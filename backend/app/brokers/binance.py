"""
Binance broker integration via CCXT async.
Supports spot trading, real-time order book, and triangular arb scanning.
Enhanced with tighter entry confirmation filters to improve signal quality.
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
    """
    Binance broker wrapper with built‑in entry confirmation filters.

    The filters aim to reduce false entries by requiring:
    • Tight bid‑ask spread
    • Minimum 24h volume
    • Price alignment with a short‑term simple moving average (SMA)
    """

    # Default filter thresholds – can be overridden via constructor arguments
    DEFAULT_MAX_SPREAD_PCT = 0.5          # maximum allowed spread in percent
    DEFAULT_MIN_VOLUME = 1000.0          # minimum 24h base volume
    DEFAULT_SMA_PERIOD = 5               # number of candles for SMA
    DEFAULT_SMA_INTERVAL = "5m"          # interval for SMA candles

    def __init__(
        self,
        api_key: str,
        secret: str,
        testnet: bool = True,
        *,
        max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
        min_volume: float = DEFAULT_MIN_VOLUME,
        sma_period: int = DEFAULT_SMA_PERIOD,
        sma_interval: str = DEFAULT_SMA_INTERVAL,
    ):
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

        # Confirmation filter parameters
        self.max_spread_pct = max_spread_pct
        self.min_volume = min_volume
        self.sma_period = sma_period
        self.sma_interval = sma_interval

    async def close(self):
        await self.exchange.close()

    # --------------------------------------------------------------------- #
    # Signal confirmation helpers
    # --------------------------------------------------------------------- #
    async def _fetch_recent_ohlcv(self, symbol: str) -> list[float]:
        """Fetch the most recent closing prices for SMA calculation."""
        tf = INTERVAL_MAP.get(self.sma_interval, "5m")
        # We request sma_period + 1 bars to ensure we have enough data even if the latest bar is incomplete
        ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=self.sma_period + 1)
        # Extract closing prices; the most recent candle is last in the list
        return [bar[4] for bar in ohlcv]

    async def _calculate_sma(self, closes: list[float]) -> float:
        """Simple moving average of the supplied closing prices."""
        if not closes:
            raise ValueError("No closing prices provided for SMA calculation")
        return sum(closes) / len(closes)

    async def _validate_entry(self, request: OrderRequest) -> None:
        """
        Perform entry confirmation checks.
        Raises BrokerError if any condition fails.
        """
        # 1. Basic quote information
        quote = await self.get_quote(request.symbol)

        # 2. Spread check
        mid_price = (quote.bid + quote.ask) / 2.0
        if mid_price == 0:
            raise BrokerError(f"Zero mid price for {request.symbol}")
        spread_pct = ((quote.ask - quote.bid) / mid_price) * 100.0
        if spread_pct > self.max_spread_pct:
            raise BrokerError(
                f"Spread {spread_pct:.2f}% exceeds max allowed {self.max_spread_pct:.2f}% for {request.symbol}"
            )

        # 3. Volume check
        if quote.volume < self.min_volume:
            raise BrokerError(
                f"24h volume {quote.volume:.2f} below minimum {self.min_volume:.2f} for {request.symbol}"
            )

        # 4. SMA alignment check
        closes = await self._fetch_recent_ohlcv(request.symbol)
        sma = await self._calculate_sma(closes)
        last_close = closes[-1]

        if request.side.lower() == "buy":
            # For a long entry we prefer price at or above SMA
            if last_close < sma:
                raise BrokerError(
                    f"Long entry price {last_close:.4f} below SMA {sma:.4f} for {request.symbol}"
                )
        elif request.side.lower() == "sell":
            # For a short entry we prefer price at or below SMA
            if last_close > sma:
                raise BrokerError(
                    f"Short entry price {last_close:.4f} above SMA {sma:.4f} for {request.symbol}"
                )
        else:
            raise BrokerError(f"Unsupported side '{request.side}' in OrderRequest")

    # --------------------------------------------------------------------- #
    # Core broker actions
    # --------------------------------------------------------------------- #
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place an order after confirming entry quality.
        """
        try:
            # Run confirmation filters before sending the order
            await self._validate_entry(request)

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
                # Fallback to market order if type is ambiguous
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
        except BrokerError:
            # Propagate our own validation errors unchanged
            raise
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