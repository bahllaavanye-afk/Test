from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import logging
import time


# Module‑level logger for structured monitoring
_logger = logging.getLogger(__name__)


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

    def __init__(self) -> None:
        """Initialize the broker base with a dedicated logger."""
        self._logger = _logger

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker. Raises BrokerError on failure."""

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> dict:
        """Get current status of an order."""

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return all open positions."""

    @abstractmethod
    async def get_account(self) -> dict:
        """Return account balance and equity."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol."""

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}."""

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------
    def _log_metrics(
        self,
        *,
        signal_count: int,
        execution_time: float,
        pnl: float | None = None,
        operation: str = "unknown",
    ) -> None:
        """
        Log structured monitoring information at INFO level.

        Parameters
        ----------
        signal_count: int
            Number of signals processed for the operation.
        execution_time: float
            Elapsed time in seconds for the operation.
        pnl: float | None
            Profit & loss realised (if applicable).
        operation: str
            Identifier for the operation being logged (e.g., "place_order").
        """
        log_payload = {
            "operation": operation,
            "signal_count": signal_count,
            "execution_time_s": round(execution_time, 6),
            "pnl": pnl,
        }
        self._logger.info("Broker monitoring metrics", extra=log_payload)

    def _measure_execution(self, func):
        """
        Decorator to measure async function execution time and emit a log entry.
        Concrete broker methods can apply this decorator to automatically log
        execution metrics.
        """
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
            # The concrete implementation should provide signal_count and pnl
            # via attributes on the result or through kwargs.
            signal_count = getattr(result, "signal_count", 0)
            pnl = getattr(result, "pnl", None)
            operation_name = func.__name__
            self._log_metrics(
                signal_count=signal_count,
                execution_time=end - start,
                pnl=pnl,
                operation=operation_name,
            )
            return result
        return wrapper