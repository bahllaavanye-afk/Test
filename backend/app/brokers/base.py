from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict
import logging

# Structured logger for broker interactions
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BrokerError(Exception):
    """Base exception for all broker-related errors."""

    def __init__(self, message: str, *, context: str | None = None, original: Exception | None = None):
        super().__init__(message)
        self.context = context
        self.original = original
        # Log the error with structured context
        logger.error(
            "BrokerError",
            extra={
                "message": message,
                "context": context,
                "original_exception": repr(original) if original else None,
            },
        )


class OrderPlacementError(BrokerError):
    """Raised when placing an order fails."""


class OrderCancellationError(BrokerError):
    """Raised when cancelling an order fails."""


class OrderRetrievalError(BrokerError):
    """Raised when retrieving an order fails."""


class PositionRetrievalError(BrokerError):
    """Raised when retrieving positions fails."""


class AccountRetrievalError(BrokerError):
    """Raised when retrieving account information fails."""


class QuoteRetrievalError(BrokerError):
    """Raised when retrieving a quote fails."""


class HistoricalDataError(BrokerError):
    """Raised when retrieving historical data fails."""


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
        raise NotImplementedError("place_order must be implemented by subclasses")

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an open order. Returns True if cancelled.

        Implementations should raise :class:`OrderCancellationError` on failure.
        """
        raise NotImplementedError("cancel_order must be implemented by subclasses")

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """
        Get current status of an order.

        Implementations should raise :class:`OrderRetrievalError` on failure.
        """
        raise NotImplementedError("get_order must be implemented by subclasses")

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Return all open positions.

        Implementations should raise :class:`PositionRetrievalError` on failure.
        """
        raise NotImplementedError("get_positions must be implemented by subclasses")

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """
        Return account balance and equity.

        Implementations should raise :class:`AccountRetrievalError` on failure.
        """
        raise NotImplementedError("get_account must be implemented by subclasses")

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """
        Return current bid/ask/last for a symbol.

        Implementations should raise :class:`QuoteRetrievalError` on failure.
        """
        raise NotImplementedError("get_quote must be implemented by subclasses")

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}.

        Implementations should raise :class:`HistoricalDataError` on failure.
        """
        raise NotImplementedError("get_historical must be implemented by subclasses")