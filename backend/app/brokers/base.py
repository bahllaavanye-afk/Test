from abc import ABC, abstractmethod
import logging
import time
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrderRequest:
    """Container for order submission parameters."""
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
    """Result returned after an order is processed by the broker."""
    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    raw_payload: dict | None = None


@dataclass(slots=True)
class QuoteResult:
    """Current market quote for a symbol."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float | None = None


class AbstractBroker(ABC):
    """Interface that all brokers must implement.

    Concrete broker implementations should call ``self._log_trade_metrics`` after
    processing a trade to emit structured logs containing:

    * ``signal_count`` – number of signals generated for the execution window
    * ``exec_time`` – elapsed time (seconds) spent handling the request
    * ``pnl`` – profit‑and‑loss realised from the trade
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the broker base class.

        The constructor is deliberately lightweight to avoid interfering with
        subclass ``__init__`` signatures. Sub‑classes may optionally call
        ``super().__init__()`` to obtain a pre‑configured logger.
        """
        self._logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _measure_execution(func):
        """Decorator to measure async function execution time and log it."""
        async def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = await func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            # ``signal_count`` and ``pnl`` are not known here; set to 0 as placeholder.
            self._log_trade_metrics(signal_count=0, exec_time=elapsed, pnl=0.0)
            return result
        return wrapper

    def _log_trade_metrics(self, *, signal_count: int, exec_time: float, pnl: float) -> None:
        """Emit a structured INFO log with key trade metrics.

        Args:
            signal_count: Number of trading signals processed.
            exec_time: Execution time in seconds for the broker operation.
            pnl: Realised profit‑and‑loss for the operation.
        """
        self._logger.info(
            "trade_metrics",
            extra={
                "signal_count": signal_count,
                "execution_time_seconds": exec_time,
                "pnl": pnl,
                "broker": self.__class__.__name__,
            },
        )

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker. Raises ``BrokerError`` on failure."""
        ...

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns ``True`` if cancelled."""
        ...

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> dict:
        """Get current status of an order."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return all open positions."""
        ...

    @abstractmethod
    async def get_account(self) -> dict:
        """Return account balance and equity."""
        ...

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol."""
        ...

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """Return OHLCV bars.

        Each dict contains: ``ts``, ``open``, ``high``, ``low``, ``close``, ``volume``.
        """
        ...