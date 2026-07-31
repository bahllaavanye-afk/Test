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
                # Fallback to market order for unsupported types
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


# ==================== Unit Tests ====================

import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestBinanceBrokerEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock the ccxt.binance class to avoid real network calls
        self.mock_exchange = MagicMock()
        self.mock_exchange.create_market_order = AsyncMock(
            return_value={"id": "123", "status": "open", "filled": 0, "average": None}
        )
        self.mock_exchange.create_limit_order = AsyncMock(
            return_value={"id": "124", "status": "open", "filled": 0, "average": None}
        )
        self.mock_exchange.fetch_ohlcv = AsyncMock(return_value=[[1609459200000, 1, 2, 0.5, 1.5, 1000]])
        self.mock_exchange.iso8601 = MagicMock(return_value="2021-01-01T00:00:00Z")
        self.mock_exchange.fetch_tickers = AsyncMock(return_value={"BTC/USDT": {"bid": 50000}})
        # Patch the ccxt.binance constructor to return our mock
        patcher = patch("ccxt.async_support.binance", return_value=self.mock_exchange)
        self.addCleanup(patcher.stop)
        self.mock_binance_ctor = patcher.start()
        self.broker = BinanceBroker(api_key="key", secret="secret", testnet=True)

    async def test_get_historical_invalid_interval_defaults_to_1d(self):
        """When an unsupported interval is supplied, BinanceBroker should default to '1d'."""
        result = await self.broker.get_historical("BTC/USDT", interval="99m")
        # Verify that fetch_ohlcv was called with the default interval '1d'
        self.mock_exchange.fetch_ohlcv.assert_awaited_once_with("BTC/USDT", "1d", limit=500)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["open"], 1)

    async def test_get_all_tickers_caching_behavior(self):
        """Second call within TTL should return cached data without invoking fetch_tickers again."""
        first = await self.broker.get_all_tickers(cache_ttl=5)
        second = await self.broker.get_all_tickers(cache_ttl=5)
        # fetch_tickers should have been called only once
        self.mock_exchange.fetch_tickers.assert_awaited_once()
        self.assertIs(first, second)  # same dict object from cache

    async def test_place_order_unknown_type_falls_back_to_market(self):
        """If an unknown order_type is provided, place_order should fallback to a market order."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.001,
            order_type="unknown_type",
            limit_price=None,
        )
        result = await self.broker.place_order(request)
        self.mock_exchange.create_market_order.assert_awaited_once_with(
            "BTC/USDT", "buy", 0.001
        )
        self.assertIsInstance(result, OrderResult)
        self.assertEqual(result.broker_order_id, "123")


if __name__ == "__main__":
    unittest.main()