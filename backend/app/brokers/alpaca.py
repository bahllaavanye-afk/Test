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
import functools
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

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


@functools.lru_cache(maxsize=1024)
def _is_crypto(symbol: str) -> bool:
    """Return True if a symbol should be routed to the crypto endpoints."""
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

        # Simple in‑memory cache for historical data to avoid duplicate requests
        self._historical_cache: Dict[Tuple[str, str, str, str], Any] = {}

    async def _call(self, fn, *args, **kwargs):
        """Throttled wrapper around blocking SDK calls."""
        async with self._limiter:
            return await asyncio.to_thread(fn, *args, **kwargs)

    # ── Orders ────────────────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order (market, limit, or bracket) and return a normalized result."""
        if request.quantity <= 0:
            raise BrokerError("Order quantity must be positive")

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
                        raw_payload={"id": str(order.id), "symbol": request.symbol, "order_class": "bracket"},
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
                    symbol=request.symbol,
                    qty=request.quantity,
                    side=side,
                    time_in_force=tif,
                )
            elif request.order_type == "limit" and request.limit_price is not None:
                req = LimitOrderRequest(
                    symbol=request.symbol,
                    qty=request.quantity,
                    side=side,
                    time_in_force=tif,
                    limit_price=request.limit_price,
                )
            else:
                raise BrokerError(f"Unsupported order type: {request.order_type}")

            order = await self._call(self.trading.submit_order, order_data=req)
            return OrderResult(
                broker_order_id=str(order.id),
                status=str(order.status),
                filled_qty=float(order.filled_qty or 0),
                avg_fill_price=(float(order.filled_avg_price) if order.filled_avg_price else None),
                raw_payload={"id": str(order.id), "symbol": request.symbol, "order_class": "plain"},
            )
        except Exception as exc:
            logger.error("Failed to place order", symbol=request.symbol, error=str(exc))
            raise BrokerError(f"Failed to place order: {exc}") from exc

    # ── Historical Data ───────────────────────────────────────────────────────

    async def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        adjust: bool = True,
    ) -> list:
        """
        Retrieve historical bar data for a symbol.

        Results are cached for the lifetime of the broker instance to avoid
        duplicate network calls when the same request is made repeatedly within
        a short time window (e.g., during back‑testing or multi‑symbol runs).
        """
        cache_key = (symbol, start.isoformat(), end.isoformat(), timeframe, str(adjust))
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        tf_obj = TF_MAP.get(timeframe)
        if not tf_obj:
            raise BrokerError(f"Unsupported timeframe: {timeframe}")

        try:
            if _is_crypto(symbol):
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    start=start,
                    end=end,
                    timeframe=tf_obj,
                    adjustment=adjust,
                )
                data = await self._call(self.crypto_data.get_crypto_bars, request)
                bars = [b._asdict() for b in data]  # Convert namedtuple to dict for consistency
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    start=start,
                    end=end,
                    timeframe=tf_obj,
                    adjustment=adjust,
                )
                data = await self._call(self.stock_data.get_stock_bars, request)
                bars = [b._asdict() for b in data]

            self._historical_cache[cache_key] = bars
            return bars
        except Exception as exc:
            logger.error("Failed to fetch historical bars", symbol=symbol, error=str(exc))
            raise BrokerError(f"Failed to fetch historical bars for {symbol}: {exc}") from exc

    # ── Latest Quote ────────────────────────────────────────────────────────

    async def get_latest_quote(self, symbol: str) -> QuoteResult:
        """Fetch the most recent quote for a symbol."""
        try:
            if _is_crypto(symbol):
                request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
                data = await self._call(self.crypto_data.get_crypto_latest_quote, request)
                quote = data[0] if isinstance(data, list) else data
            else:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                data = await self._call(self.stock_data.get_stock_latest_quote, request)
                quote = data[0] if isinstance(data, list) else data

            return QuoteResult(
                symbol=symbol,
                bid_price=float(quote.bid_price) if quote.bid_price is not None else None,
                ask_price=float(quote.ask_price) if quote.ask_price is not None else None,
                timestamp=quote.timestamp.replace(tzinfo=timezone.utc) if quote.timestamp else None,
                raw_payload=quote._asdict(),
            )
        except Exception as exc:
            logger.error("Failed to fetch latest quote", symbol=symbol, error=str(exc))
            raise BrokerError(f"Failed to fetch latest quote for {symbol}: {exc}") from exc

    # ── Positions & Cash ─────────────────────────────────────────────────────

    async def get_positions(self) -> list:
        """Return a list of current positions."""
        try:
            positions = await self._call(self.trading.get_all_positions)
            return [p._asdict() for p in positions]
        except Exception as exc:
            logger.error("Failed to fetch positions", error=str(exc))
            raise BrokerError(f"Failed to fetch positions: {exc}") from exc

    async def get_cash(self) -> float:
        """Return the current cash balance."""
        try:
            account = await self._call(self.trading.get_account)
            return float(account.cash)
        except Exception as exc:
            logger.error("Failed to fetch cash balance", error=str(exc))
            raise BrokerError(f"Failed to fetch cash balance: {exc}") from exc

    # ── Order Management ────────────────────────────────────────────────────

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order by its broker‑provided identifier."""
        try:
            await self._call(self.trading.cancel_order, broker_order_id)
            return True
        except Exception as exc:
            logger.warning("Failed to cancel order", broker_order_id=broker_order_id, error=str(exc))
            return False

    async def list_orders(self, status: str = "open") -> list:
        """List orders filtered by status."""
        try:
            query_status = QueryOrderStatus(status.upper())
            request = GetOrdersRequest(status=query_status)
            orders = await self._call(self.trading.get_orders, request)
            return [o._asdict() for o in orders]
        except Exception as exc:
            logger.error("Failed to list orders", error=str(exc))
            raise BrokerError(f"Failed to list orders: {exc}") from exc