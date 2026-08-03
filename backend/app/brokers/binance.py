"""
Binance broker integration via CCXT async.
Supports spot trading, real-time order book, and triangular arb scanning.
"""
import asyncio
import time
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    ccxt = None  # type: ignore
    CCXT_AVAILABLE = False
    logger.info("ccxt not installed — Binance broker disabled")


INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class BinanceBroker(AbstractBroker):
    # Configuration thresholds for entry validation
    _SPREAD_THRESHOLD = 0.005  # 0.5%
    _MIN_VOLUME = 10.0         # Minimum base asset volume
    _CONFIRMATION_CANDLES = 3  # Number of candles to confirm trend

    def __init__(self, api_key: str, secret: str, testnet: bool = True):
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret,
                "options": {"defaultType": "spot"},
                "enableRateLimit": True,
                "timeout": 30000,
            }
        )
        if testnet:
            self.exchange.set_sandbox_mode(True)

        # Cache for expensive calls
        self._ticker_cache = {"data": None, "timestamp": 0.0}
        self._ticker_lock = asyncio.Lock()

    async def close(self):
        await self.exchange.close()

    async def _fetch_quote(self, symbol: str) -> QuoteResult:
        """Internal helper to fetch a fresh quote with timeout handling."""
        try:
            ticker = await asyncio.wait_for(
                self.exchange.fetch_ticker(symbol), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Binance fetch_ticker timed out", symbol=symbol)
            raise BrokerError(f"Binance quote timed out for {symbol}")

        return QuoteResult(
            symbol=symbol,
            bid=float(ticker["bid"]),
            ask=float(ticker["ask"]),
            last=float(ticker["last"]),
            volume=float(ticker.get("baseVolume", 0)),
        )

    async def _validate_entry_conditions(
        self, request: OrderRequest
    ) -> None:
        """
        Tighten entry conditions:
        1. Ensure spread is within acceptable threshold.
        2. Verify minimum volume.
        3. Confirm short‑term price trend aligns with order side.
        """
        quote = await self._fetch_quote(request.symbol)

        # 1. Spread check
        spread = (quote.ask - quote.bid) / quote.bid if quote.bid > 0 else 1.0
        if spread > self._SPREAD_THRESHOLD:
            raise BrokerError(
                f"Spread {spread:.4%} exceeds threshold for {request.symbol}"
            )

        # 2. Volume check
        if quote.volume < self._MIN_VOLUME:
            raise BrokerError(
                f"Volume {quote.volume:.2f} below minimum for {request.symbol}"
            )

        # 3. Trend confirmation using recent candles
        ohlc = await self.get_historical(
            request.symbol, interval="5m", limit=self._CONFIRMATION_CANDLES
        )
        if len(ohlc) < self._CONFIRMATION_CANDLES:
            raise BrokerError(
                f"Insufficient historical data for confirmation on {request.symbol}"
            )

        # Determine direction of recent closes
        closes = [float(bar["close"]) for bar in ohlc]
        trend_up = all(earlier < later for earlier, later in zip(closes, closes[1:]))
        trend_down = all(earlier > later for earlier, later in zip(closes, closes[1:]))

        if request.side == "buy" and not trend_up:
            raise BrokerError(f"Buy entry trend not confirmed for {request.symbol}")
        if request.side == "sell" and not trend_down:
            raise BrokerError(f"Sell entry trend not confirmed for {request.symbol}")

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place an order after validating entry conditions.
        Market orders are default; limit orders require a limit_price.
        """
        # Entry validation – any failure raises BrokerError, preventing the order.
        await self._validate_entry_conditions(request)

        try:
            if request.order_type == "market":
                order = await self.exchange.create_market_order(
                    request.symbol, request.side, request.quantity
                )
            elif request.order_type == "limit" and request.limit_price:
                order = await self.exchange.create_limit_order(
                    request.symbol,
                    request.side,
                    request.quantity,
                    request.limit_price,
                )
            else:
                # Fallback to market order if limit specifics are missing
                order = await self.exchange.create_market_order(
                    request.symbol, request.side, request.quantity
                )

            return OrderResult(
                broker_order_id=str(order["id"]),
                status=order["status"],
                filled_qty=float(order.get("filled", 0)),
                avg_fill_price=float(order["average"])
                if order.get("average")
                else None,
                raw_payload=order,
            )
        except Exception as e:
            raise BrokerError(f"Binance: {e}")

    async def cancel_order(self, broker_order_id: str, symbol: str = "") -> bool:
        try:
            await self.exchange.cancel_order(broker_order_id, symbol)
            return True
        except Exception as e:
            logger.warning(
                "Binance cancel_order failed",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str, symbol: str = "") -> dict:
        return await self.exchange.fetch_order(broker_order_id, symbol)

    async def get_positions(self) -> list[dict]:
        balance = await self.exchange.fetch_balance()
        positions = []
        for asset, info in balance["total"].items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        return positions

    async def get_account(self) -> dict:
        balance = await self.exchange.fetch_balance()
        usdt = balance["total"].get("USDT", 0)
        return {
            "equity": usdt,
            "cash": usdt,
            "buying_power": usdt,
            "portfolio_value": usdt,
        }

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Public wrapper that reuses the internal quote fetcher."""
        return await self._fetch_quote(symbol)

    async def get_historical(
        self, symbol: str, interval: str = "1d", limit: int = 500
    ) -> list[dict]:
        tf = INTERVAL_MAP.get(interval, "1d")
        ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        return [
            {
                "ts": self.exchange.iso8601(bar[0]),
                "open": bar[1],
                "high": bar[2],
                "low": bar[3],
                "close": bar[4],
                "volume": bar[5],
            }
            for bar in ohlcv
        ]

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self.exchange.fetch_order_book(symbol, limit)

    async def get_all_tickers(self, cache_ttl: int = 30) -> dict:
        """Fetch all tickers for triangular arb scanning with simple TTL caching."""
        async with self._ticker_lock:
            now = time.monotonic()
            if (
                self._ticker_cache["data"] is not None
                and now - self._ticker_cache["timestamp"] < cache_ttl
            ):
                return self._ticker_cache["data"]
            try:
                data = await self.exchange.fetch_tickers()
                self._ticker_cache.update({"data": data, "timestamp": now})
                return data
            except Exception as e:
                logger.error("Failed to fetch tickers from Binance", error=str(e))
                raise BrokerError(f"Binance ticker fetch error: {e}")