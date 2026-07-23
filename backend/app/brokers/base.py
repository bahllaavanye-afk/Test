from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import logging
import time


# Configure a module‑level logger; concrete broker implementations can adjust handlers/formatters as needed.
_logger = logging.getLogger("quantedge.broker")
_logger.setLevel(logging.INFO)


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

    # ---------------------------------------------------------------------
    # Monitoring utilities
    # ---------------------------------------------------------------------
    def _log_trade_metrics(self, signal_count: int, exec_time_seconds: float, pnl: float) -> None:
        """
        Emit a structured INFO‑level log entry containing key trade‑related metrics.

        Parameters
        ----------
        signal_count: int
            Number of signals processed for the current batch or strategy run.
        exec_time_seconds: float
            Total execution time (in seconds) for the operation being logged.
        pnl: float
            Profit‑and‑loss realized (positive for profit, negative for loss).
        """
        # Using the ``extra`` dict enables downstream log processors to treat the
        # fields as structured data rather than a plain string.
        _logger.info(
            "Trade metrics",
            extra={
                "signal_count": signal_count,
                "execution_time_seconds": exec_time_seconds,
                "pnl": pnl,
            },
        )
    # End of class AbstractBroker