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
from typing import Any, Dict, List, Optional

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
    from alpaca.trading.errors import APIError as TradingAPIError, OrderNotFoundError
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestQuoteRequest,
        CryptoBarsRequest,
        CryptoLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.errors import APIError as DataAPIError
    ALPACA_AVAILABLE = True
except ImportError:  # pragma: no cover
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — Alpaca broker unavailable")

# Bracket order support — imported lazily so missing symbols don't break the module
try:
    from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
    ALPACA_BRACKET_AVAILABLE = True
except ImportError:  # pragma: no cover
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
    """Return True if *symbol* refers to a crypto pair."""
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
    except Exception as exc:  # pragma: no cover
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

    async def _call(self, fn, *args, **kwargs):
        """Throttled wrapper around blocking SDK calls with error handling."""
        async with self._limiter:
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except (TradingAPIError, DataAPIError) as api_err:
                logger.error(
                    "Alpaca API error",
                    function=fn.__name__,
                    error=str(api_err),
                    args=args,
                    kwargs=kwargs,
                )
                raise BrokerError(f"Alpaca API error: {api_err}") from api_err
            except Exception as exc:  # pragma: no cover
                logger.exception(
                    "Unexpected error during Alpaca SDK call",
                    function=fn.__name__,
                    args=args,
                    kwargs=kwargs,
                )
                raise BrokerError(f"Unexpected Alpaca SDK error: {exc}") from exc

    # ── Orders ────────────────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order on Alpaca.

        Handles market, limit, and bracket orders. Errors from the Alpaca SDK
        are caught and re‑raised as :class:`BrokerError` with structured logging.
        """
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
                        raw_payload={
                            "id": str(order.id),
                            "symbol": request.symbol,
                            "order_class": "bracket",
                        },
                    )
                except BrokerError:
                    # Propagate already‑logged BrokerError
                    raise
                except Exception as bracket_exc:  # pragma: no cover
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

            logger.info(
                "Submitting order",
                symbol=request.symbol,
                order_type=request.order_type,
                quantity=request.quantity,
            )
            order = await self._call(self.trading.submit_order, order_data=req)

            return OrderResult(
                broker_order_id=str(order.id),
                status=str(order.status),
                filled_qty=float(order.filled_qty or 0),
                avg_fill_price=(
                    float(order.filled_avg_price) if order.filled_avg_price else None
                ),
                raw_payload={"id": str(order.id), "symbol": request.symbol},
            )
        except BrokerError:
            # Already logged in _call or above; just re‑raise
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to place order", symbol=request.symbol, error=str(exc))
            raise BrokerError(f"Failed to place order: {exc}") from exc

    async def get_order(self, broker_order_id: str) -> OrderResult:
        """Retrieve order status by broker order ID."""
        try:
            order = await self._call(self.trading.get_order, broker_order_id)
            return OrderResult(
                broker_order_id=str(order.id),
                status=str(order.status),
                filled_qty=float(order.filled_qty or 0),
                avg_fill_price=(
                    float(order.filled_avg_price) if order.filled_avg_price else None
                ),
                raw_payload={"id": str(order.id), "symbol": order.symbol},
            )
        except OrderNotFoundError as not_found:
            logger.warning(
                "Order not found",
                broker_order_id=broker_order_id,
                error=str(not_found),
            )
            raise BrokerError(f"Order {broker_order_id} not found") from not_found
        except BrokerError:
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to retrieve order", broker_order_id=broker_order_id)
            raise BrokerError(f"Failed to retrieve order: {exc}") from exc

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            await self._call(self.trading.cancel_order, broker_order_id)
            logger.info("Cancelled order", broker_order_id=broker_order_id)
            return True
        except OrderNotFoundError as not_found:
            logger.warning(
                "Cancel failed – order not found",
                broker_order_id=broker_order_id,
                error=str(not_found),
            )
            raise BrokerError(f"Order {broker_order_id} not found") from not_found
        except BrokerError:
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to cancel order", broker_order_id=broker_order_id)
            raise BrokerError(f"Failed to cancel order: {exc}") from exc

    # ── Market Data ────────────────────────────────────────────────────────────

    async def get_latest_quote(self, symbol: str) -> QuoteResult:
        """Fetch the latest quote for *symbol*."""
        try:
            if _is_crypto(symbol):
                request = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
                resp = await self._call(self.crypto_data.get_latest_quote, request)
                quote = resp[0] if resp else None
            else:
                request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
                resp = await self._call(self.stock_data.get_latest_quote, request)
                quote = resp[0] if resp else None

            if not quote:
                raise BrokerError(f"No quote returned for {symbol}")

            return QuoteResult(
                bid_price=float(quote.bid_price) if hasattr(quote, "bid_price") else None,
                ask_price=float(quote.ask_price) if hasattr(quote, "ask_price") else None,
                bid_size=float(quote.bid_size) if hasattr(quote, "bid_size") else None,
                ask_size=float(quote.ask_size) if hasattr(quote, "ask_size") else None,
                timestamp=quote.timestamp,
                raw_payload=quote,
            )
        except BrokerError:
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to fetch latest quote", symbol=symbol)
            raise BrokerError(f"Failed to fetch latest quote for {symbol}: {exc}") from exc

    async def get_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical bar data for *symbol*."""
        try:
            tf = TF_MAP.get(timeframe)
            if not tf:
                raise BrokerError(f"Unsupported timeframe: {timeframe}")

            if _is_crypto(symbol):
                request = CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    start=start,
                    end=end,
                    timeframe=tf,
                    limit=limit,
                )
                bars = await self._call(self.crypto_data.get_crypto_bars, request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    start=start,
                    end=end,
                    timeframe=tf,
                    limit=limit,
                )
                bars = await self._call(self.stock_data.get_stock_bars, request)

            result = []
            for bar in bars:
                result.append(
                    {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    }
                )
            return result
        except BrokerError:
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception(
                "Failed to fetch historical data",
                symbol=symbol,
                timeframe=timeframe,
                error=str(exc),
            )
            raise BrokerError(f"Failed to fetch history for {symbol}: {exc}") from exc

    # Additional methods (e.g., get_positions, get_account) would follow the same
    # error‑handling pattern, converting Alpaca‑specific exceptions into
    # ``BrokerError`` and logging them with structured context.