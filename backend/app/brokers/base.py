from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from typing import Any, List, Dict, Optional

logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Base exception for all broker‑related errors."""

    def __init__(self, message: str, *, operation: str = "", payload: Any = None):
        super().__init__(message)
        self.operation = operation
        self.payload = payload


class OrderPlacementError(BrokerError):
    """Raised when an order cannot be placed."""


class OrderCancellationError(BrokerError):
    """Raised when an order cannot be cancelled."""


class OrderRetrievalError(BrokerError):
    """Raised when order information cannot be retrieved."""


class PositionRetrievalError(BrokerError):
    """Raised when positions cannot be retrieved."""


class AccountRetrievalError(BrokerError):
    """Raised when account information cannot be retrieved."""


class QuoteRetrievalError(BrokerError):
    """Raised when a quote cannot be retrieved."""


class HistoricalDataError(BrokerError):
    """Raised when historical data cannot be fetched."""


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
    raw_payload: Optional[Dict] = None


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
        """
        Submit an order to the broker.

        Raises
        ------
        OrderPlacementError
            If the order cannot be placed.
        """
        raise NotImplementedError("place_order must be implemented by a subclass")

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an open order.

        Returns
        -------
        bool
            True if the order was successfully cancelled.

        Raises
        ------
        OrderCancellationError
            If the cancellation fails.
        """
        raise NotImplementedError("cancel_order must be implemented by a subclass")

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict:
        """
        Get current status of an order.

        Raises
        ------
        OrderRetrievalError
            If the order information cannot be fetched.
        """
        raise NotImplementedError("get_order must be implemented by a subclass")

    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """
        Return all open positions.

        Raises
        ------
        PositionRetrievalError
            If positions cannot be retrieved.
        """
        raise NotImplementedError("get_positions must be implemented by a subclass")

    @abstractmethod
    async def get_account(self) -> Dict:
        """
        Return account balance and equity.

        Raises
        ------
        AccountRetrievalError
            If account information cannot be retrieved.
        """
        raise NotImplementedError("get_account must be implemented by a subclass")

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """
        Return current bid/ask/last for a symbol.

        Raises
        ------
        QuoteRetrievalError
            If the quote cannot be retrieved.
        """
        raise NotImplementedError("get_quote must be implemented by a subclass")

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict]:
        """
        Return OHLCV bars.

        Each dict contains: {ts, open, high, low, close, volume}.

        Raises
        ------
        HistoricalDataError
            If historical data cannot be fetched.
        """
        raise NotImplementedError("get_historical must be implemented by a subclass")

    # --------------------------------------------------------------------- #
    # Helper methods for structured error logging
    # --------------------------------------------------------------------- #
    def _log_error(self, operation: str, exc: Exception) -> None:
        """
        Log an exception with structured details.

        Parameters
        ----------
        operation: str
            Name of the operation being performed.
        exc: Exception
            The caught exception.
        """
        logger.error(
            "Broker operation failed",
            extra={
                "operation": operation,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            exc_info=True,
        )

    # --------------------------------------------------------------------- #
    # Safe wrappers that catch broker‑specific errors and log them.
    # Sub‑classes can use these wrappers to avoid repetitive try/except blocks.
    # --------------------------------------------------------------------- #
    async def place_order_safe(self, request: OrderRequest) -> Optional[OrderResult]:
        try:
            return await self.place_order(request)
        except OrderPlacementError as e:
            self._log_error("place_order", e)
        except BrokerError as e:
            self._log_error("place_order", e)
        return None

    async def cancel_order_safe(self, broker_order_id: str) -> bool:
        try:
            return await self.cancel_order(broker_order_id)
        except OrderCancellationError as e:
            self._log_error("cancel_order", e)
        except BrokerError as e:
            self._log_error("cancel_order", e)
        return False

    async def get_order_safe(self, broker_order_id: str) -> Optional[Dict]:
        try:
            return await self.get_order(broker_order_id)
        except OrderRetrievalError as e:
            self._log_error("get_order", e)
        except BrokerError as e:
            self._log_error("get_order", e)
        return None

    async def get_positions_safe(self) -> List[Dict]:
        try:
            return await self.get_positions()
        except PositionRetrievalError as e:
            self._log_error("get_positions", e)
        except BrokerError as e:
            self._log_error("get_positions", e)
        return []

    async def get_account_safe(self) -> Optional[Dict]:
        try:
            return await self.get_account()
        except AccountRetrievalError as e:
            self._log_error("get_account", e)
        except BrokerError as e:
            self._log_error("get_account", e)
        return None

    async def get_quote_safe(self, symbol: str) -> Optional[QuoteResult]:
        try:
            return await self.get_quote(symbol)
        except QuoteRetrievalError as e:
            self._log_error("get_quote", e)
        except BrokerError as e:
            self._log_error("get_quote", e)
        return None

    async def get_historical_safe(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict]:
        try:
            return await self.get_historical(symbol, interval, limit)
        except HistoricalDataError as e:
            self._log_error("get_historical", e)
        except BrokerError as e:
            self._log_error("get_historical", e)
        return []