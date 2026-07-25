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
            logger.warning("Binance cancel_order failed", order_id=broker_order_id, symbol=symbol, error=str(e))
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
# Unit Tests for Edge Cases
# ==============================
import pytest
import unittest.mock

@pytest.fixture
def mock_exchange():
    """Create an async mock mimicking the ccxt exchange interface."""
    mock = unittest.mock.AsyncMock()
    mock.fetch_ohlcv = unittest.mock.AsyncMock()
    mock.fetch_tickers = unittest.mock.AsyncMock()
    mock.fetch_ticker = unittest.mock.AsyncMock()
    mock.iso8601 = unittest.mock.Mock(side_effect=lambda ts: f"ISO{ts}")
    return mock

@pytest.fixture
def broker(mock_exchange):
    """Instantiate BinanceBroker with a mocked exchange."""
    b = BinanceBroker(api_key="test_key", secret="test_secret", testnet=False)
    b.exchange = mock_exchange
    return b


@pytest.mark.asyncio
async def test_get_historical_invalid_interval_fallback(broker, mock_exchange):
    """When an unknown interval is supplied, BinanceBroker should fall back to '1d'."""
    mock_exchange.fetch_ohlcv.return_value = []
    await broker.get_historical("BTC/USDT", interval="invalid_interval")
    mock_exchange.fetch_ohlcv.assert_awaited_once()
    # The second argument should be the default '1d' mapping
    args = mock_exchange.fetch_ohlcv.call_args[0]
    assert args[1] == "1d"


@pytest.mark.asyncio
async def test_get_all_tickers_caching_behavior(broker, mock_exchange):
    """Within the cache TTL, subsequent calls should return cached data without invoking fetch_tickers."""
    mock_exchange.fetch_tickers.return_value = {"BTC/USDT": {"bid": 50000}}
    first_result = await broker.get_all_tickers(cache_ttl=5)
    second_result = await broker.get_all_tickers(cache_ttl=5)
    assert first_result is second_result  # Same object indicates caching
    assert mock_exchange.fetch_tickers.await_count == 1  # Called only once


@pytest.mark.asyncio
async def test_get_quote_timeout_raises_broker_error(broker, mock_exchange):
    """A timeout while fetching a ticker should be translated into a BrokerError."""
    mock_exchange.fetch_ticker.side_effect = asyncio.TimeoutError
    with pytest.raises(BrokerError) as exc_info:
        await broker.get_quote("ETH/USDT")
    assert "quote timed out" in str(exc_info.value).lower()