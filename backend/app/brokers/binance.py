"""
Binance broker integration via CCXT async.
Supports spot trading, real-time order book, and triangular arb scanning.
"""
import asyncio
import time
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
    def __init__(self, api_key: str, secret: str, testnet: bool = True):
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non‑empty string")
        if not isinstance(secret, str) or not secret:
            raise ValueError("secret must be a non‑empty string")
        if not isinstance(testnet, bool):
            raise ValueError("testnet must be a boolean")
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

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if not isinstance(request, OrderRequest):
            raise ValueError("request must be an OrderRequest instance")
        if not isinstance(request.symbol, str) or not request.symbol:
            raise ValueError("request.symbol must be a non‑empty string")
        if request.side not in {"buy", "sell"}:
            raise ValueError("request.side must be either 'buy' or 'sell'")
        if request.order_type not in {"market", "limit"}:
            raise ValueError("request.order_type must be 'market' or 'limit'")
        if not isinstance(request.quantity, (int, float)) or request.quantity <= 0:
            raise ValueError("request.quantity must be a positive number")
        if request.order_type == "limit":
            if request.limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            if not isinstance(request.limit_price, (int, float)) or request.limit_price <= 0:
                raise ValueError("request.limit_price must be a positive number")

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
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")
        if not isinstance(symbol, str):
            raise ValueError("symbol must be a string")
        try:
            await self.exchange.cancel_order(broker_order_id, symbol)
            return True
        except Exception:
            return False

    async def get_order(self, broker_order_id: str, symbol: str = "") -> dict:
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")
        if not isinstance(symbol, str):
            raise ValueError("symbol must be a string")
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
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")
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
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")
        if interval not in INTERVAL_MAP:
            raise ValueError(f"interval must be one of {list(INTERVAL_MAP.keys())}")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
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
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return await self.exchange.fetch_order_book(symbol, limit)

    async def get_all_tickers(self, cache_ttl: int = 30) -> dict:
        """Fetch all tickers for triangular arb scanning with simple TTL caching."""
        if not isinstance(cache_ttl, int) or cache_ttl <= 0:
            raise ValueError("cache_ttl must be a positive integer")
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