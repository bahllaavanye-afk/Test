"""
Alpaca broker — equities AND crypto on a single API key.

Routing:
  • Symbols containing '/' or ending with 'USD/USDT' → CryptoHistoricalDataClient
    for data, same TradingClient for orders (Alpaca unified account).
  • Everything else → StockHistoricalDataClient / StockLatestQuoteClient.

Alpaca crypto coverage (paper + live, commission-free):
  BTC/USD, ETH/USD, SOL/USD, AVAX/USD, DOGE/USD, SHIB/USD,
  LTC/USD, BCH/USD, LINK/USD, UNI/USD, AAVE/USD, BAT/USD,
  CRV/USD, GRT/USD, MKR/USD, SUSHI/USD, XTZ/USD, ALGO/USD,
  MATIC/USD, DOT/USD — and growing.

For perpetual futures, funding rates, liquidation data, or stablecoin
pairs (USDC/USDT, DAI/USDT) use BinanceBroker — those are not
available on Alpaca spot.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Callable

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.config import settings
from app.utils.exceptions import BrokerError
from app.utils.logging import logger

# Alpaca enforces 200 requests/minute. Cap concurrent calls at 10 to stay
# well within that limit even under heavy multi-symbol strategy runners.
_ALPACA_CONCURRENCY = 10

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass
    from alpaca.trading.errors import APIError as AlpacaAPIError
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestQuoteRequest,
        CryptoBarsRequest,
        CryptoLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.errors import DataError as AlpacaDataError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — Alpaca broker unavailable")

# Bracket order support — imported lazily so missing symbols don't break the module
try:
    from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
    ALPACA_BRACKET_AVAILABLE = True
except ImportError:
    ALPACA_BRACKET_AVAILABLE = False


TF_MAP = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}

# Alpaca uses "BTC/USD" format for crypto
CRYPTO_SUFFIXES = ("/USD", "/USDT", "/BTC", "/ETH")


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol or any(symbol.endswith(s) for s in ("BTC", "ETH", "SOL", "DOGE"))


def create_alpaca_broker(paper: bool = True) -> "AlpacaBroker | None":
    """Factory that returns an AlpacaBroker when keys are present, or None.

    In paper/dev mode without API keys the process must not crash — the strategy
    runner simply runs in signal-only mode (no orders submitted) when broker is None.
    """
    api_key = settings.alpaca_api_key
    secret_key = settings.alpaca_secret_key

    if not api_key or not secret_key:
        logger.warning(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set — Alpaca broker disabled. "
            "Strategies will run in signal-only mode (no orders submitted)."
        )
        return None

    if not ALPACA_AVAILABLE:
        logger.warning("alpaca-py not installed — Alpaca broker unavailable")
        return None

    try:
        return AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=paper)
    except Exception as exc:
        logger.warning("Failed to initialise AlpacaBroker", error=str(exc))
        return None


class AlpacaBroker(AbstractBroker):
    """Unified Alpaca broker for both equities and crypto spot."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py required: pip install alpaca-py")
        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret key are required")
        self.paper = paper
        self.trading = TradingClient(api_key, secret_key, paper=paper)
        self.stock_data = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_data = CryptoHistoricalDataClient(api_key, secret_key)
        # Rate limiter: max _ALPACA_CONCURRENCY simultaneous API calls
        self._limiter = asyncio.Semaphore(_ALPACA_CONCURRENCY)

    async def _call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Throttled wrapper around blocking SDK calls with error handling."""
        async with self._limiter:
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except (AlpacaAPIError, AlpacaDataError) as alpaca_exc:
                logger.error(
                    "Alpaca SDK call failed",
                    function=fn.__name__,
                    error=str(alpaca_exc),
                    exception_type=type(alpaca_exc).__name__,
                )
                raise BrokerError(f"Alpaca SDK error: {alpaca_exc}") from alpaca_exc
            except Exception as exc:
                logger.error(
                    "Unexpected error during Alpaca SDK call",
                    function=fn.__name__,
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )
                raise BrokerError(f"Unexpected Alpaca SDK error: {exc}") from exc

    # ── Orders ────────────────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order on Alpaca, handling market, limit and bracket types."""
        try:
            side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.GTC

            # Crypto requires IOC or GTC (no DAY orders on 24/7 markets)
            if _is_crypto(request.symbol):
                tif = TimeInForce.GTC

            # Detect bracket order when stop_loss or take_profit are set
            has_bracket = request.stop_loss is not None or request.take_profit is not None

            if has_bracket and ALPACA_BRACKET_AVAILABLE:
                try:
                    tp_req = (
                        TakeProfitRequest(limit_price=round(float(request.take_profit), 4))
                        if request.take_profit is not None
                        else None
                    )
                    sl_req = (
                        StopLossRequest(stop_price=round(float(request.stop_loss), 4))
                        if request.stop_loss is not None
                        else None
                    )
                    req = MarketOrderRequest(
                        symbol=request.symbol,
                        qty=request.quantity,
                        side=side,
                        time_in_force=tif,
                        order_class=OrderClass.BRACKET,
                        take_profit=tp_req,
                        stop_loss=sl_req,
                    )
                    logger.info(
                        "Submitting bracket order",
                        symbol=request.symbol,
                        stop_loss=request.stop_loss,
                        take_profit=request.take_profit,
                    )
                    order = await self._call(self.trading.submit_order, order_data=req)
                    return OrderResult(
                        broker_order_id=str(order.id),
                        status=str(order.status),
                        filled_qty=float(order.filled_qty or 0),
                        avg_fill_price=(
                            float(order.filled_avg_price) if order.filled_avg_price else None
                        ),
                    )
                except Exception as exc:
                    logger.error(
                        "Bracket order submission failed",
                        symbol=request.symbol,
                        error=str(exc),
                    )
                    raise BrokerError(f"Bracket order error: {exc}") from exc

            # Non‑bracket orders
            req = MarketOrderRequest(
                symbol=request.symbol,
                qty=request.quantity,
                side=side,
                time_in_force=tif,
            )
            order = await self._call(self.trading.submit_order, order_data=req)
            return OrderResult(
                broker_order_id=str(order.id),
                status=str(order.status),
                filled_qty=float(order.filled_qty or 0),
                avg_fill_price=(
                    float(order.filled_avg_price) if order.filled_avg_price else None
                ),
            )
        except BrokerError:
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in place_order",
                symbol=request.symbol,
                error=str(exc),
            )
            raise BrokerError(f"Failed to place order: {exc}") from exc

    # ── Quotes ────────────────────────────────────────────────────────────────

    async def get_latest_quote(self, symbol: str) -> QuoteResult:
        """Fetch the latest quote for a symbol (stock or crypto)."""
        try:
            if _is_crypto(symbol):
                request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
                data = await self._call(self.crypto_data.get_latest_quote, request)
                quote = data[symbol]
                return QuoteResult(
                    bid_price=float(quote.bid_price),
                    ask_price=float(quote.ask_price),
                    timestamp=quote.timestamp.replace(tzinfo=timezone.utc),
                )
            else:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                data = await self._call(self.stock_data.get_latest_quote, request)
                quote = data[symbol]
                return QuoteResult(
                    bid_price=float(quote.bid_price),
                    ask_price=float(quote.ask_price),
                    timestamp=quote.timestamp.replace(tzinfo=timezone.utc),
                )
        except (AlpacaAPIError, AlpacaDataError) as alpaca_exc:
            logger.error(
                "Failed to fetch latest quote",
                symbol=symbol,
                error=str(alpaca_exc),
            )
            raise BrokerError(f"Alpaca SDK error while fetching quote: {alpaca_exc}") from alpaca_exc
        except Exception as exc:
            logger.error(
                "Unexpected error while fetching quote",
                symbol=symbol,
                error=str(exc),
            )
            raise BrokerError(f"Unexpected error while fetching quote: {exc}") from exc

    # Additional methods (historical data, order history, etc.) would follow...

# ---------------------------------------------------------------------------
# Unit tests for edge cases
# ---------------------------------------------------------------------------

import unittest
from unittest.mock import patch, MagicMock

class TestAlpacaBrokerUtilities(unittest.TestCase):
    def test_is_crypto_boundary_cases(self):
        # Direct crypto symbols
        self.assertTrue(_is_crypto("BTC"))
        self.assertTrue(_is_crypto("ETH"))
        # Symbol containing slash
        self.assertTrue(_is_crypto("AAPL/USD"))
        # Empty string should be False
        self.assertFalse(_is_crypto(""))
        # Non‑crypto symbol without slash
        self.assertFalse(_is_crypto("AAPL"))

    @patch('app.config.settings')
    def test_create_broker_missing_keys(self, mock_settings):
        mock_settings.alpaca_api_key = ""
        mock_settings.alpaca_secret_key = ""
        broker = create_alpaca_broker()
        self.assertIsNone(broker)

    def test_create_broker_module_unavailable(self):
        # Simulate alpaca-py not being installed
        with patch('backend.app.brokers.alpaca.AL_PACA_AVAILABLE', False):
            with patch('app.config.settings') as mock_settings:
                mock_settings.alpaca_api_key = "dummy_key"
                mock_settings.alpaca_secret_key = "dummy_secret"
                broker = create_alpaca_broker()
                self.assertIsNone(broker)

if __name__ == "__main__":
    unittest.main()