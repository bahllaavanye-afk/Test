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


# ==============================
# Unit tests for edge conditions
# ==============================
import pytest

class _AsyncMock:
    """Simple async mock helper."""
    def __init__(self, return_value=None, side_effect=None):
        self._return = return_value
        self._side_effect = side_effect
        self.called_with = []

    async def __call__(self, *args, **kwargs):
        self.called_with.append((args, kwargs))
        if self._side_effect:
            raise self._side_effect
        return self._return

    async def fetch_ohlcv(self, symbol, tf, limit):
        self.called_with.append(("fetch_ohlcv", symbol, tf, limit))
        return [(0, 1, 2, 3, 4, 5)]

    async def fetch_tickers(self):
        self.called_with.append(("fetch_tickers",))
        return {"BTC/USDT": {"bid": 50000, "ask": 50010}}

    async def fetch_ticker(self, symbol):
        self.called_with.append(("fetch_ticker", symbol))
        return {"bid": 1, "ask": 2, "last": 1.5, "baseVolume": 1000}


@pytest.fixture
def mock_broker():
    """Create a BinanceBroker with a mocked exchange."""
    broker = BinanceBroker(api_key="test", secret="test", testnet=True)
    broker.exchange = type("MockExchange", (), {})()
    broker.exchange.create_market_order = _AsyncMock(return_value={"id": "1", "status": "closed", "filled": 0.5, "average": 100})
    broker.exchange.create_limit_order = _AsyncMock(return_value={"id": "2", "status": "open", "filled": 0, "average": None})
    broker.exchange.fetch_balance = _AsyncMock(return_value={"total": {"USDT": 1000, "BTC": 0.1}})
    broker.exchange.fetch_order = _AsyncMock(return_value={"id": "1", "status": "closed"})
    broker.exchange.fetch_order_book = _AsyncMock(return_value={"bids": [], "asks": []})
    broker.exchange.fetch_ticker = _AsyncMock(return_value={"bid": 10, "ask": 11, "last": 10.5, "baseVolume": 200})
    broker.exchange.fetch_ohlcv = _AsyncMock(return_value=[(1622505600000, 100, 110, 90, 105, 1500)])
    broker.exchange.fetch_tickers = _AsyncMock(return_value={"BTC/USDT": {"bid": 50000, "ask": 50010}})
    broker.exchange.iso8601 = lambda ts: f"{ts}"
    broker.exchange.set_sandbox_mode = lambda x: None
    return broker


@pytest.mark.asyncio
async def test_get_historical_interval_fallback(mock_broker):
    """When an unsupported interval is supplied, BinanceBroker should fallback to '1d'."""
    await mock_broker.get_historical(symbol="BTC/USDT", interval="unsupported")
    # Verify that fetch_ohlcv was called with the default timeframe '1d'
    called = mock_broker.exchange.fetch_ohlcv.called_with
    assert any(call[1] == "1d" for call in called), "fetch_ohlcv not called with fallback interval '1d'"


@pytest.mark.asyncio
async def test_get_all_tickers_caching(mock_broker):
    """Cache should return the same data within TTL and avoid a second fetch_tickers call."""
    first = await mock_broker.get_all_tickers(cache_ttl=5)
    second = await mock_broker.get_all_tickers(cache_ttl=5)
    # The mock should have been called only once
    fetch_calls = [c for c in mock_broker.exchange.fetch_tickers.called_with if c[0] == "fetch_tickers"]
    assert len(fetch_calls) == 1, "fetch_tickers called more than once within cache TTL"
    assert first is second, "Cached ticker data not returned on second call"


@pytest.mark.asyncio
async def test_get_quote_timeout_handling():
    """If fetch_ticker exceeds the timeout, a BrokerError should be raised."""
    broker = BinanceBroker(api_key="test", secret="test", testnet=True)
    # Replace exchange with a mock that sleeps longer than the timeout
    async def slow_fetch_ticker(symbol):
        await asyncio.sleep(0.2)  # longer than the 0.1s timeout we will set
        return {"bid": 1, "ask": 2, "last": 1.5, "baseVolume": 100}
    broker.exchange = type("MockExchange", (), {})()
    broker.exchange.fetch_ticker = slow_fetch_ticker
    broker.exchange.set_sandbox_mode = lambda x: None

    # Patch the timeout to a very short value to force timeout
    original_wait_for = asyncio.wait_for
    async def short_wait_for(coro, timeout):
        return await original_wait_for(coro, timeout=0.1)
    asyncio.wait_for = short_wait_for

    with pytest.raises(BrokerError) as exc:
        await broker.get_quote("BTC/USDT")
    assert "timed out" in str(exc.value)

    # Restore original wait_for to avoid side effects
    asyncio.wait_for = original_wait_for