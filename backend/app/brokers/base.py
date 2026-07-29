from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, TypeVar
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Exception hierarchy for broker operations
class BrokerError(Exception):
    """Base exception for all broker related errors."""

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}

class OrderPlacementError(BrokerError):
    """Raised when an order cannot be placed."""

class OrderCancellationError(BrokerError):
    """Raised when an order cannot be cancelled."""

class OrderRetrievalError(BrokerError):
    """Raised when order details cannot be fetched."""

class PositionRetrievalError(BrokerError):
    """Raised when positions cannot be fetched."""

class AccountRetrievalError(BrokerError):
    """Raised when account information cannot be fetched."""

class QuoteRetrievalError(BrokerError):
    """Raised when a quote cannot be fetched."""

class HistoricalDataError(BrokerError):
    """Raised when historical data cannot be retrieved."""

# Generic type for coroutine return values
_T = TypeVar("_T")

def _log_exception(exc: Exception, *, context: dict | None = None) -> None:
    """
    Centralised logging for exceptions with optional contextual data.
    """
    if context:
        logger.exception("%s | Context: %s", exc, context)
    else:
        logger.exception("%s", exc)

async def _handle_coroutine_error(
    coro: Coroutine[Any, Any, _T],
    error_cls: TypeVar("BrokerError", bound=BrokerError),
    *,
    context: dict | None = None,
) -> _T:
    """
    Executes a coroutine, logs any exception, and raises a typed BrokerError.

    Parameters
    ----------
    coro: Coroutine
        The coroutine to execute.
    error_cls: BrokerError subclass
        The specific error type to raise on failure.
    context: dict, optional
        Additional information to include in logs for debugging.

    Returns
    -------
    The result of the coroutine if successful.
    """
    try:
        return await coro
    except Exception as exc:  # pragma: no cover
        _log_exception(exc, context=context)
        raise error_cls(str(exc), context=context) from exc


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: str               # buy|sell
    order_type: str         # market|limit|stop|bracket
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    stop_loss: float | None = None      # for bracket orders
    take_profit: float | None = None    # for bracket orders
    time_in_force: str = "GTC"
    account_id: str = ""
    strategy_id: str | None = None
    risk_bucket: str = "directional"   # for risk manager routing
    execution_algo: str = "limit_first"  # market|limit_first|twap|vwap


@dataclass(slots=True)
class OrderResult:
    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    raw_payload: dict | None = None


@dataclass(slots=True)
class QuoteResult:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float | None = None


class AbstractBroker(ABC):
    """Interface that all brokers must implement."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit an order to the broker.

        Implementations should raise :class:`OrderPlacementError` on failure.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an open order.

        Returns True if cancelled. Implementations should raise
        :class:`OrderCancellationError` on failure.
        """

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> dict:
        """
        Get current status of an order.

        Implementations should raise :class:`OrderRetrievalError` on failure.
        """

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """
        Return all open positions.

        Implementations should raise :class:`PositionRetrievalError` on failure.
        """

    @abstractmethod
    async def get_account(self) -> dict:
        """
        Return account balance and equity.

        Implementations should raise :class:`AccountRetrievalError` on failure.
        """

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """
        Return current bid/ask/last for a symbol.

        Implementations should raise :class:`QuoteRetrievalError` on failure.
        """

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """
        Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}.

        Implementations should raise :class:`HistoricalDataError` on failure.
        """

    # -------------------------------------------------------------------------
    # Helper methods for concrete brokers
    # -------------------------------------------------------------------------

    async def _execute_place_order(
        self, request: OrderRequest, coro: Coroutine[Any, Any, OrderResult]
    ) -> OrderResult:
        """Execute ``place_order`` coroutine with unified error handling."""
        return await _handle_coroutine_error(
            coro,
            OrderPlacementError,
            context={"request": request.__dict__},
        )

    async def _execute_cancel_order(
        self, broker_order_id: str, coro: Coroutine[Any, Any, bool]
    ) -> bool:
        """Execute ``cancel_order`` coroutine with unified error handling."""
        return await _handle_coroutine_error(
            coro,
            OrderCancellationError,
            context={"broker_order_id": broker_order_id},
        )

    async def _execute_get_order(
        self, broker_order_id: str, coro: Coroutine[Any, Any, dict]
    ) -> dict:
        """Execute ``get_order`` coroutine with unified error handling."""
        return await _handle_coroutine_error(
            coro,
            OrderRetrievalError,
            context={"broker_order_id": broker_order_id},
        )

    async def _execute_get_positions(
        self, coro: Coroutine[Any, Any, list[dict]]
    ) -> list[dict]:
        """Execute ``get_positions`` coroutine with unified error handling."""
        return await _handle_coroutine_error(
            coro,
            PositionRetrievalError,
        )

    async def _execute_get_account(
        self, coro: Coroutine[Any, Any, dict]
    ) -> dict:
        """Execute ``get_account`` coroutine with unified error handling."""
        return await _handle_coroutine_error(
            coro,
            AccountRetrievalError,
        )

    async def _execute_get_quote(
        self, symbol: str, coro: Coroutine[Any, Any, QuoteResult]
    ) -> QuoteResult:
        """Execute ``get_quote`` coroutine with unified error handling."""
        return await _handle_coroutine_error(
            coro,
            QuoteRetrievalError,
            context={"symbol": symbol},
        )

    async def _execute_get_historical(
        self,
        symbol: str,
        interval: str,
        limit: int,
        coro: Coroutine[Any, Any, list[dict]],
    ) -> list[dict]:
        """Execute ``get_historical`` coroutine with unified error handling."""
        return await _handle_coroutine_error(
            coro,
            HistoricalDataError,
            context={"symbol": symbol, "interval": interval, "limit": limit},
        )