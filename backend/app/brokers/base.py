from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import logging


logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Base exception for all broker related errors."""

    def __init__(self, message: str, *, original_exception: Optional[BaseException] = None):
        super().__init__(message)
        self.original_exception = original_exception


class OrderPlacementError(BrokerError):
    """Raised when an order cannot be placed."""


class OrderCancellationError(BrokerError):
    """Raised when an order cannot be cancelled."""


class OrderRetrievalError(BrokerError):
    """Raised when an order cannot be retrieved."""


class PositionsRetrievalError(BrokerError):
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
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class QuoteResult:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: Optional[float] = None


class AbstractBroker(ABC):
    """Interface that all brokers must implement with robust error handling."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit an order to the broker.

        Raises:
            OrderPlacementError: If the order cannot be placed.
        """
        try:
            raise NotImplementedError("place_order must be implemented by the subclass.")
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to place order for %s: %s",
                request.symbol,
                exc,
                exc_info=True,
            )
            raise OrderPlacementError(
                f"Failed to place order for {request.symbol}", original_exception=exc
            ) from exc

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an open order.

        Returns:
            bool: True if cancelled.

        Raises:
            OrderCancellationError: If cancellation fails.
        """
        try:
            raise NotImplementedError("cancel_order must be implemented by the subclass.")
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to cancel order %s: %s",
                broker_order_id,
                exc,
                exc_info=True,
            )
            raise OrderCancellationError(
                f"Failed to cancel order {broker_order_id}", original_exception=exc
            ) from exc

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """
        Get current status of an order.

        Raises:
            OrderRetrievalError: If the order cannot be retrieved.
        """
        try:
            raise NotImplementedError("get_order must be implemented by the subclass.")
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to retrieve order %s: %s",
                broker_order_id,
                exc,
                exc_info=True,
            )
            raise OrderRetrievalError(
                f"Failed to retrieve order {broker_order_id}", original_exception=exc
            ) from exc

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Return all open positions.

        Raises:
            PositionsRetrievalError: If positions cannot be retrieved.
        """
        try:
            raise NotImplementedError("get_positions must be implemented by the subclass.")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to retrieve positions: %s", exc, exc_info=True)
            raise PositionsRetrievalError(
                "Failed to retrieve positions", original_exception=exc
            ) from exc

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """
        Return account balance and equity.

        Raises:
            AccountRetrievalError: If account info cannot be retrieved.
        """
        try:
            raise NotImplementedError("get_account must be implemented by the subclass.")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to retrieve account information: %s", exc, exc_info=True)
            raise AccountRetrievalError(
                "Failed to retrieve account information", original_exception=exc
            ) from exc

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """
        Return current bid/ask/last for a symbol.

        Raises:
            QuoteRetrievalError: If the quote cannot be retrieved.
        """
        try:
            raise NotImplementedError("get_quote must be implemented by the subclass.")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to retrieve quote for %s: %s", symbol, exc, exc_info=True)
            raise QuoteRetrievalError(
                f"Failed to retrieve quote for {symbol}", original_exception=exc
            ) from exc

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Return OHLCV bars.

        Each dict: {ts, open, high, low, close, volume}.

        Raises:
            HistoricalDataError: If historical data cannot be fetched.
        """
        try:
            raise NotImplementedError("get_historical must be implemented by the subclass.")
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to retrieve historical data for %s (%s): %s",
                symbol,
                interval,
                exc,
                exc_info=True,
            )
            raise HistoricalDataError(
                f"Failed to retrieve historical data for {symbol} ({interval})",
                original_exception=exc,
            ) from exc