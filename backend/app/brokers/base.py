from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict
import logging


# Configure module logger
logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Base exception for all broker related errors."""


class OrderPlacementError(BrokerError):
    """Raised when an order cannot be placed."""


class OrderCancellationError(BrokerError):
    """Raised when an order cannot be cancelled."""


class OrderRetrievalError(BrokerError):
    """Raised when order details cannot be retrieved."""


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

        Raises:
            OrderPlacementError: If the order cannot be placed.
        """
        logger.error(
            "place_order not implemented",
            extra={"request": request, "exception": "OrderPlacementError"},
        )
        raise OrderPlacementError("place_order method must be implemented by subclass")

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an open order.

        Returns:
            bool: True if cancelled.

        Raises:
            OrderCancellationError: If the cancellation fails.
        """
        logger.error(
            "cancel_order not implemented",
            extra={"broker_order_id": broker_order_id, "exception": "OrderCancellationError"},
        )
        raise OrderCancellationError("cancel_order method must be implemented by subclass")

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict:
        """
        Get current status of an order.

        Raises:
            OrderRetrievalError: If the order information cannot be retrieved.
        """
        logger.error(
            "get_order not implemented",
            extra={"broker_order_id": broker_order_id, "exception": "OrderRetrievalError"},
        )
        raise OrderRetrievalError("get_order method must be implemented by subclass")

    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """
        Return all open positions.

        Raises:
            PositionRetrievalError: If positions cannot be retrieved.
        """
        logger.error(
            "get_positions not implemented",
            extra={"exception": "PositionRetrievalError"},
        )
        raise PositionRetrievalError("get_positions method must be implemented by subclass")

    @abstractmethod
    async def get_account(self) -> Dict:
        """
        Return account balance and equity.

        Raises:
            AccountRetrievalError: If account information cannot be retrieved.
        """
        logger.error(
            "get_account not implemented",
            extra={"exception": "AccountRetrievalError"},
        )
        raise AccountRetrievalError("get_account method must be implemented by subclass")

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """
        Return current bid/ask/last for a symbol.

        Raises:
            QuoteRetrievalError: If the quote cannot be retrieved.
        """
        logger.error(
            "get_quote not implemented",
            extra={"symbol": symbol, "exception": "QuoteRetrievalError"},
        )
        raise QuoteRetrievalError("get_quote method must be implemented by subclass")

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict]:
        """
        Return OHLCV bars.

        Each dict: {ts, open, high, low, close, volume}.

        Raises:
            HistoricalDataError: If historical data cannot be fetched.
        """
        logger.error(
            "get_historical not implemented",
            extra={
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "exception": "HistoricalDataError",
            },
        )
        raise HistoricalDataError("get_historical method must be implemented by subclass")