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
from typing import List, Optional

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
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestQuoteRequest,
        CryptoBarsRequest,
        CryptoLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — Alpaca broker unavailable")

# Bracket order support — imported lazily so missing symbols don't break the module
try:
    from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
    from alpaca.trading.enums import OrderClass
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


def _validate_order_request(request: OrderRequest) -> None:
    """Validate an OrderRequest instance."""
    if not isinstance(request, OrderRequest):
        raise ValueError("request must be an OrderRequest instance")

    if not isinstance(request.symbol, str) or not request.symbol.strip():
        raise ValueError("order symbol must be a non‑empty string")

    if request.quantity is None:
        raise ValueError("order quantity must be provided")
    try:
        qty = float(request.quantity)
    except Exception:
        raise ValueError("order quantity must be numeric")
    if qty <= 0:
        raise ValueError("order quantity must be positive")

    if not isinstance(request.side, str) or request.side.lower() not in ("buy", "sell"):
        raise ValueError("order side must be either 'buy' or 'sell'")

    valid_order_types = {"market", "moc", "limit"}
    if not isinstance(request.order_type, str) or request.order_type.lower() not in valid_order_types:
        raise ValueError(f"order_type must be one of {sorted(valid_order_types)}")

    if request.order_type.lower() == "limit":
        if request.limit_price is None:
            raise ValueError("limit_price must be provided for limit orders")
        try:
            lp = float(request.limit_price)
        except Exception:
            raise ValueError("limit_price must be numeric")
        if lp <= 0:
            raise ValueError("limit_price must be positive")

    if request.stop_loss is not None:
        try:
            sl = float(request.stop_loss)
        except Exception:
            raise ValueError("stop_loss must be numeric")
        if sl <= 0:
            raise ValueError("stop_loss must be positive")

    if request.take_profit is not None:
        try:
            tp = float(request.take_profit)
        except Exception:
            raise ValueError("take_profit must be numeric")
        if tp <= 0:
            raise ValueError("take_profit must be positive")


def _validate_symbol(symbol: str) -> None:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non‑empty string")


def _validate_timeframe(tf: str) -> None:
    if not isinstance(tf, str) or tf not in TF_MAP:
        raise ValueError(f"timeframe must be one of {sorted(TF_MAP.keys())}")


def _validate_datetime_range(start: datetime, end: datetime) -> None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise ValueError("start and end must be datetime objects")
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if start >= end:
        raise ValueError("start datetime must be earlier than end datetime")


def create_alpaca_broker(paper: bool = True) -> "AlpacaBroker | None":
    """Factory that returns an AlpacaBroker when keys are present, or None.

    In paper/dev mode without API keys the process must not crash — the strategy
    runner simply runs in signal‑only mode (no orders submitted) when broker is None.
    """
    if not isinstance(paper, bool):
        raise ValueError("paper flag must be a boolean")

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
        if not isinstance(paper, bool):
            raise ValueError("paper flag must be a boolean")

        self.paper = paper
        self.trading = TradingClient(api_key, secret_key, paper=paper)
        self.stock_data = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_data = CryptoHistoricalDataClient(api_key, secret_key)
        # Rate limiter: max _ALPACA_CONCURRENCY simultaneous API calls
        self._limiter = asyncio.Semaphore(_ALPACA_CONCURRENCY)

    async def _call(self, fn, *args, **kwargs):
        """Throttled wrapper around blocking SDK calls."""
        async with self._limiter:
            return await asyncio.to_thread(fn, *args, **kwargs)

    # ── Orders ────────────────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to Alpaca."""
        _validate_order_request(request)

        try:
            side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.GTC

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

            if request.order_type.lower() in ("market", "moc"):
                req = MarketOrderRequest(
                    symbol=request.symbol,
                    qty=request.quantity,
                    side=side,
                    time_in_force=tif,
                )
            elif request.order_type.lower() == "limit" and request.limit_price is not None:
                req = LimitOrderRequest(
                    symbol=request.symbol,
                    qty=request.quantity,
                    side=side,
                    time_in_force=tif,
                    limit_price=request.limit_price,
                )
            else:
                raise ValueError(f"Unsupported order_type: {request.order_type}")

            logger.info("Submitting order", symbol=request.symbol, order_type=request.order_type)
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
            raise BrokerError(f"Failed to place order for {request.symbol}: {exc}") from exc

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order."""
        if not isinstance(broker_order_id, str) or not broker_order_id.strip():
            raise ValueError("broker_order_id must be a non‑empty string")
        try:
            await self._call(self.trading.cancel_order, broker_order_id=broker_order_id)
            logger.info("Cancelled order", broker_order_id=broker_order_id)
            return True
        except Exception as exc:
            raise BrokerError(f"Failed to cancel order {broker_order_id}: {exc}") from exc

    async def get_orders(self, status: str = "open") -> List[OrderResult]:
        """Retrieve orders filtered by status."""
        valid_statuses = {"open", "closed", "all"}
        if not isinstance(status, str) or status.lower() not in valid_statuses:
            raise ValueError(f"status must be one of {sorted(valid_statuses)}")
        try:
            req = GetOrdersRequest(status=QueryOrderStatus(status.upper()))
            orders = await self._call(self.trading.get_orders, request=req)
            return [
                OrderResult(
                    broker_order_id=str(o.id),
                    status=str(o.status),
                    filled_qty=float(o.filled_qty or 0),
                    avg_fill_price=(float(o.filled_avg_price) if o.filled_avg_price else None),
                    raw_payload={"id": str(o.id), "symbol": o.symbol},
                )
                for o in orders
            ]
        except Exception as exc:
            raise BrokerError(f"Failed to retrieve orders: {exc}") from exc

    # ── Market Data ─────────────────────────────────────────────────────────────

    async def get_latest_quote(self, symbol: str) -> QuoteResult:
        """Fetch the latest quote for a given symbol."""
        _validate_symbol(symbol)

        try:
            if _is_crypto(symbol):
                req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = await self._call(self.crypto_data.get_crypto_latest_quote, request=req)
                data = quote[symbol]
                return QuoteResult(
                    symbol=symbol,
                    bid_price=float(data.bid_price) if data.bid_price else None,
                    ask_price=float(data.ask_price) if data.ask_price else None,
                    timestamp=data.timestamp,
                    raw_payload=data,
                )
            else:
                req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = await self._call(self.stock_data.get_stock_latest_quote, request=req)
                data = quote[symbol]
                return QuoteResult(
                    symbol=symbol,
                    bid_price=float(data.bid_price) if data.bid_price else None,
                    ask_price=float(data.ask_price) if data.ask_price else None,
                    timestamp=data.timestamp,
                    raw_payload=data,
                )
        except Exception as exc:
            raise BrokerError(f"Failed to fetch latest quote for {symbol}: {exc}") from exc