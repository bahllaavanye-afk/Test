from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import logging


logger = logging.getLogger("quantedge.broker")


class BrokerError(Exception):
    """Base exception for all broker‑related errors."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


class OrderPlacementError(BrokerError):
    """Raised when an order cannot be placed."""


class OrderCancellationError(BrokerError):
    """Raised when an order cannot be cancelled."""


class OrderRetrievalError(BrokerError):
    """Raised when an order cannot be retrieved."""


class PositionRetrievalError(BrokerError):
    """Raised when positions cannot be retrieved."""


class AccountRetrievalError(BrokerError):
    """Raised when account information cannot be retrieved."""


class QuoteRetrievalError(BrokerError):
    """Raised when a quote cannot be retrieved."""


class HistoricalDataError(BrokerError):
    """Raised when historical data cannot be retrieved."""


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
        """Submit an order to the broker. Implementations should raise
        :class:`OrderPlacementError` on failure.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Implementations should raise
        :class:`OrderCancellationError` on failure.
        """

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict:
        """Get current status of an order. Implementations should raise
        :class:`OrderRetrievalError` on failure.
        """

    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Return all open positions. Implementations should raise
        :class:`PositionRetrievalError` on failure.
        """

    @abstractmethod
    async def get_account(self) -> Dict:
        """Return account balance and equity. Implementations should raise
        :class:`AccountRetrievalError` on failure.
        """

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol. Implementations should raise
        :class:`QuoteRetrievalError` on failure.
        """

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict]:
        """Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}.
        Implementations should raise :class:`HistoricalDataError` on failure.
        """

    # -------------------------------------------------------------------------
    # Helper methods providing structured error handling and logging
    # -------------------------------------------------------------------------

    async def safe_place_order(self, request: OrderRequest) -> OrderResult:
        try:
            return await self.place_order(request)
        except Exception as exc:
            self._log_error(exc, "place_order")
            raise OrderPlacementError("Failed to place order", exc) from exc

    async def safe_cancel_order(self, broker_order_id: str) -> bool:
        try:
            return await self.cancel_order(broker_order_id)
        except Exception as exc:
            self._log_error(exc, "cancel_order")
            raise OrderCancellationError("Failed to cancel order", exc) from exc

    async def safe_get_order(self, broker_order_id: str) -> Dict:
        try:
            return await self.get_order(broker_order_id)
        except Exception as exc:
            self._log_error(exc, "get_order")
            raise OrderRetrievalError("Failed to retrieve order", exc) from exc

    async def safe_get_positions(self) -> List[Dict]:
        try:
            return await self.get_positions()
        except Exception as exc:
            self._log_error(exc, "get_positions")
            raise PositionRetrievalError("Failed to retrieve positions", exc) from exc

    async def safe_get_account(self) -> Dict:
        try:
            return await self.get_account()
        except Exception as exc:
            self._log_error(exc, "get_account")
            raise AccountRetrievalError("Failed to retrieve account info", exc) from exc

    async def safe_get_quote(self, symbol: str) -> QuoteResult:
        try:
            return await self.get_quote(symbol)
        except Exception as exc:
            self._log_error(exc, "get_quote")
            raise QuoteRetrievalError(f"Failed to retrieve quote for {symbol}", exc) from exc

    async def safe_get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict]:
        try:
            return await self.get_historical(symbol, interval, limit)
        except Exception as exc:
            self._log_error(exc, "get_historical")
            raise HistoricalDataError(
                f"Failed to retrieve historical data for {symbol}", exc
            ) from exc

    @staticmethod
    def _log_error(error: Exception, operation: str) -> None:
        logger.error(
            "Broker operation error",
            extra={
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )