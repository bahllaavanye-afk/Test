from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Base exception for all broker‑related errors."""

    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class BrokerConnectionError(BrokerError):
    """Raised when a connection to the broker cannot be established or is lost."""


class BrokerResponseError(BrokerError):
    """Raised when the broker returns an unexpected or malformed response."""


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: str               # buy|sell
    order_type: str         # market|limit|stop|bracket
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None      # for bracket orders
    take_profit: Optional[float] = None    # for bracket orders
    time_in_force: str = "GTC"
    account_id: str = ""
    strategy_id: Optional[str] = None
    risk_bucket: str = "directional"   # for risk manager routing
    execution_algo: str = "limit_first"  # market|limit_first|twap|vwap


@dataclass(slots=True)
class OrderResult:
    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class QuoteResult:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: Optional[float] = None


class AbstractBroker(ABC):
    """Interface that all brokers must implement."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker.

        Raises:
            BrokerError: If the order cannot be placed.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order.

        Returns:
            bool: True if the order was successfully cancelled.

        Raises:
            BrokerError: If the cancellation fails.
        """

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Get current status of an order.

        Raises:
            BrokerError: If the order information cannot be retrieved.
        """

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions.

        Raises:
            BrokerError: If positions cannot be fetched.
        """

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """Return account balance and equity.

        Raises:
            BrokerError: If account information cannot be retrieved.
        """

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol.

        Raises:
            BrokerError: If the quote cannot be obtained.
        """

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Return OHLCV bars.

        Each dict contains: {ts, open, high, low, close, volume}.

        Raises:
            BrokerError: If historical data cannot be fetched.
        """

    # --------------------------------------------------------------------- #
    # Helper utilities for concrete implementations
    # --------------------------------------------------------------------- #

    def _log_error(self, method: str, exc: Exception, **context: Any) -> None:
        """Log an error in a structured way.

        Args:
            method: Name of the method where the error occurred.
            exc: The caught exception.
            **context: Additional key‑value pairs providing context (e.g., symbol,
                order_id, etc.).
        """
        logger.error(
            "Error in %s: %s",
            method,
            str(exc),
            exc_info=True,
            extra={"method": method, "exception": type(exc).__name__, **context},
        )