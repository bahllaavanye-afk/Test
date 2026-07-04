"""
Binance broker integration via CCXT async.
Supports spot trading, real-time order book, and triangular arb scanning.
"""
import asyncio
import time
import unittest
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
        if CCXT_AVAILABLE:
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
        else:
            # Allow instantiation for testing purposes when ccxt is unavailable.
            self.exchange = None
            if testnet:
                logger.warning("Testnet mode requested but ccxt is unavailable.")

        # Cache for expensive calls
        self._ticker_cache = {"data": None, "timestamp": 0.0}
        self._ticker_lock = asyncio.Lock()

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if not self.exchange:
            raise BrokerError("Binance exchange not initialized.")
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
        if not self.exchange:
            return False
        try:
            await self.exchange.cancel_order(broker_order_id, symbol)
            return True
        except Exception:
            return False

    async def get_order(self, broker_order_id: str, symbol: str = "") -> dict:
        if not self.exchange:
            raise BrokerError("Binance exchange not initialized.")
        return await self.exchange.fetch_order(broker_order_id, symbol)

    async def get_positions(self) -> list[dict]:
        if not self.exchange:
            raise BrokerError("Binance exchange not initialized.")
        balance = await self.exchange.fetch_balance()
        positions = []
        for asset, info in balance["total"].items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        return positions

    async def get_account(self) -> dict:
        if not self.exchange:
            raise BrokerError("Binance exchange not initialized.")
        balance = await self.exchange.fetch_balance()
        usdt = balance["total"].get("USDT", 0)
        return {
            "equity": usdt,
            "cash": usdt,
            "buying_power": usdt,
            "portfolio_value": usdt,
        }

    async def get_quote(self, symbol: str) -> QuoteResult:
        if not self.exchange:
            raise BrokerError("Binance exchange not initialized.")
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
        if not self.exchange:
            raise BrokerError("Binance exchange not initialized.")
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
        if not self.exchange:
            raise BrokerError("Binance exchange not initialized.")
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
            if not self.exchange:
                raise BrokerError("Binance exchange not initialized.")
            try:
                data = await self.exchange.fetch_tickers()
                self._ticker_cache.update({"data": data, "timestamp": now})
                return data
            except Exception as e:
                logger.error("Failed to fetch tickers from Binance", error=str(e))
                raise BrokerError(f"Binance ticker fetch error: {e}")


# ==========================
# Unit tests for edge cases
# ==========================
class _MockExchange:
    """Simple async mock exchange used for unit tests."""

    def __init__(self):
        self.last_tf = None
        self.fetch_tickers_called = 0

    async def fetch_ohlcv(self, symbol, tf, limit):
        self.last_tf = tf
        return []  # empty data set

    async def fetch_ticker(self, symbol):
        return {"bid": "1", "ask": "2", "last": "1.5", "baseVolume": "100"}

    async def fetch_tickers(self):
        self.fetch_tickers_called += 1
        return {"BTC/USDT": {}}

    async def close(self):
        pass

    def iso8601(self, ts):
        return str(ts)


class _TimeoutMockExchange:
    """Mock exchange that triggers a timeout."""

    async def fetch_ticker(self, symbol):
        raise asyncio.TimeoutError()


class TestBinanceBroker(unittest.IsolatedAsyncioTestCase):
    async def test_get_historical_invalid_interval_fallback(self):
        broker = BinanceBroker("key", "secret", testnet=True)
        broker.exchange = _MockExchange()
        await broker.get_historical("BTC/USDT", interval="invalid_interval")
        self.assertEqual(broker.exchange.last_tf, "1d")

    async def test_get_all_tickers_caching(self):
        broker = BinanceBroker("key", "secret", testnet=True)
        mock = _MockExchange()
        broker.exchange = mock
        data_first = await broker.get_all_tickers(cache_ttl=1)
        data_second = await broker.get_all_tickers(cache_ttl=10)
        self.assertIs(data_first, data_second)
        self.assertEqual(mock.fetch_tickers_called, 1)

    async def test_get_quote_timeout_raises(self):
        broker = BinanceBroker("key", "secret", testnet=True)
        broker.exchange = _TimeoutMockExchange()
        with self.assertRaises(BrokerError):
            await broker.get_quote("BTC/USDT")