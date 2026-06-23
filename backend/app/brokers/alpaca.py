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
from functools import lru_cache
from typing import Dict

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.config import settings
from app.utils.exceptions import BrokerError
from app.utils.logging import logger

# Alpaca enforces 200 requests/minute. Cap concurrent calls at 10 to stay
# well within that limit even under heavy multi-symbol strategy runners.
_ALPACA_CONCURRENCY = 10

try:
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import (
        CryptoBarsRequest,
        CryptoLatestQuoteRequest,
        StockBarsRequest,
        StockLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
        StopOrderRequest,
    )
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — Alpaca broker unavailable")

# Bracket order support — imported lazily so missing symbols don't break the module
try:
    from alpaca.trading.enums import OrderClass
    from alpaca.trading.requests import StopLossRequest, TakeProfitRequest
    ALPACA_BRACKET_AVAILABLE = True
except ImportError:
    ALPACA_BRACKET_AVAILABLE = False


TF_MAP: Dict[str, TimeFrame] = {
    "1m":  TimeFrame(1,  TimeFrameUnit.Minute),
    "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
    "4h":  TimeFrame(4,  TimeFrameUnit.Hour),
    "1d":  TimeFrame(1,  TimeFrameUnit.Day),
}

# Alpaca uses "BTC/USD" format for crypto
CRYPTO_SUFFIXES = ("/USD", "/USDT", "/BTC", "/ETH")


@lru_cache(maxsize=None)
def _is_crypto(symbol: str) -> bool:
    """Check if symbol is a crypto symbol."""
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
        self.trading     = TradingClient(api_key, secret_key, paper=paper)
        self.stock_data  = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_data = CryptoHistoricalDataClient(api_key, secret_key)
        # Rate limiter: max _ALPACA_CONCURRENCY simultaneous API calls
        self._limiter = asyncio.Semaphore(_ALPACA_CONCURRENCY)

    async def _call(self, fn, *args, **kwargs):
        """Throttled wrapper around blocking SDK calls."""
        async with self._limiter:
            return await asyncio.to_thread(fn, *args, **kwargs)

    # ── Orders ────────────────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResult:
        try:
            side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL
            tif  = TimeInForce.GTC

            # Crypto requires IOC or GTC (no DAY orders on 24/7 markets)
            if _is_crypto(request.symbol):
                tif = TimeInForce.GTC

            # Detect bracket order when stop_loss or take_profit are set
            has_bracket = (request.stop_loss is not None or request.take_profit is not None)

            if has_bracket and ALPACA_BRACKET_AVAILABLE:
                try:
                    tp_req = (
                        TakeProfitRequest(limit_price=round(float(request.take_profit), 4))
                        if request.take_profit is not None else None
                    )
                    sl_req = (
                        StopLossRequest(stop_price=round(float(request.stop_loss), 4))
                        if request.stop_loss is not None else None
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
                        avg_fill_price=(float(order.filled_avg_price)
                                        if order.filled_avg_price else None),
                        raw_payload={"id": str(order.id), "symbol": request.symbol,
                                     "order_class": "bracket"},
                    )
                except Exception as bracket_exc:
                    logger.warning(
                        "Bracket order failed — falling back to plain market order",
                        symbol=request.symbol,
                        error=str(bracket_exc),
                    )
                    # Fall through to plain order below

            if request.order_type in ("market", "moc"):
                req = MarketOrderRequest(
                    symbol=request.symbol, qty=request.quantity,
                    side=side, time_in_force=tif,
                )
            elif request.order_type == "limit" and request.limit_price:
                req = LimitOrderRequest(
                    symbol=request.symbol, qty=request.quantity,
                    side=side, time_in_force=tif,
                    limit_price=request.limit_price,
                )
            elif request.order_type == "stop" and request.stop_price:
                req = StopOrderRequest(
                    symbol=request.symbol, qty=request.quantity,
                    side=side, time_in_force=tif,
                    stop_price=request.stop_price,
                )
            else:
                raise ValueError(f"Invalid order type: {request.order_type}")

            logger.info(
                "Submitting order",
                symbol=request.symbol,
                order_type=request.order_type,
                side=request.side,
                qty=request.quantity,
            )
            order = await self._call(self.trading.submit_order, order_data=req)
            return OrderResult(
                broker_order_id=str(order.id),
                status=str(order.status),
                filled_qty=float(order.filled_qty or 0),
                avg_fill_price=(float(order.filled_avg_price)
                                if order.filled_avg_price else None),
                raw_payload={"id": str(order.id), "symbol": request.symbol},
            )
        except Exception as exc:
            logger.error("Failed to place order", error=str(exc))
            raise BrokerError("Failed to place order") from exc