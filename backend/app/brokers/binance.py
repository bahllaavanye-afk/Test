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


# ----------------------------------------------------------------------
# Unit tests for edge cases (boundary conditions)
# ----------------------------------------------------------------------
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestBinanceBrokerEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Patch the ccxt.binance constructor to return a mock exchange
        self.exchange_mock = MagicMock()
        self.exchange_mock.set_sandbox_mode = MagicMock()
        self.exchange_mock.fetch_ohlcv = AsyncMock(return_value=[])
        self.exchange_mock.fetch_tickers = AsyncMock(return_value={"BTC/USDT": {}})
        self.exchange_mock.fetch_ticker = AsyncMock(return_value={"bid": "50000", "ask": "50010", "last": "50005", "baseVolume": "100"})
        self.exchange_mock.iso8601 = MagicMock(side_effect=lambda ts: f"2022-01-01T00:00:{ts}Z")
        patcher = patch("ccxt.async_support.binance", return_value=self.exchange_mock)
        self.addCleanup(patcher.stop)
        self.mock_binance_ctor = patcher.start()

        self.broker = BinanceBroker(api_key="test", secret="test", testnet=True)

    async def test_get_historical_interval_fallback(self):
        """When an unsupported interval is passed, BinanceBroker should fallback to '1d'."""
        await self.broker.get_historical(symbol="BTC/USDT", interval="invalid_interval")
        # Verify that fetch_ohlcv was called with the default timeframe '1d'
        self.exchange_mock.fetch_ohlcv.assert_awaited_once_with("BTC/USDT", "1d", limit=500)

    async def test_get_all_tickers_cache_boundary(self):
        """Cache should be used if within TTL and refreshed after TTL expires."""
        # First call populates the cache
        first_data = await self.broker.get_all_tickers(cache_ttl=1)
        self.exchange_mock.fetch_tickers.assert_awaited_once()
        self.exchange_mock.fetch_tickers.reset_mock()

        # Second call within TTL should hit cache, no additional fetch
        second_data = await self.broker.get_all_tickers(cache_ttl=1)
        self.exchange_mock.fetch_tickers.assert_not_awaited()
        self.assertIs(first_data, second_data)

        # Advance time beyond TTL and ensure cache refreshes
        original_time = time.monotonic

        def fake_monotonic():
            return original_time() + 2  # exceed ttl of 1 second

        with patch("time.monotonic", fake_monotonic):
            await self.broker.get_all_tickers(cache_ttl=1)
        self.exchange_mock.fetch_tickers.assert_awaited_once()

    async def test_get_quote_timeout_raises_broker_error(self):
        """A timeout while fetching a ticker should raise BrokerError."""
        # Simulate a long-running fetch_ticker that exceeds the timeout
        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(0.2)  # longer than the 0.1s timeout we will set
        self.exchange_mock.fetch_ticker = AsyncMock(side_effect=slow_fetch)

        # Temporarily reduce the wait timeout to trigger the condition quickly
        with patch.object(self.broker, "get_quote", wraps=self.broker.get_quote) as wrapped:
            original_timeout = 10.0
            # monkey‑patch asyncio.wait_for inside get_quote via a context manager
            async def limited_get_quote(symbol):
                try:
                    ticker = await asyncio.wait_for(
                        self.exchange_mock.fetch_ticker(symbol), timeout=0.05
                    )
                except asyncio.TimeoutError:
                    raise BrokerError(f"Binance quote timed out for {symbol}")
            wrapped.side_effect = limited_get_quote

            with self.assertRaises(BrokerError) as cm:
                await self.broker.get_quote("BTC/USDT")
            self.assertIn("timed out", str(cm.exception))


if __name__ == "__main__":
    unittest.main()