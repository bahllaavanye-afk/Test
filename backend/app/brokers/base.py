from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import logging


logger = logging.getLogger(__name__)


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

    # --------------------------------------------------------------------- #
    # Helper validation methods (internal use)
    # --------------------------------------------------------------------- #

    def _validate_order_request(self, request: OrderRequest) -> None:
        """Validate fields of an OrderRequest.

        Raises:
            ValueError: If any field is invalid.
        """
        if not request.symbol:
            raise ValueError("OrderRequest.symbol must be non‑empty")
        if request.side not in {"buy", "sell"}:
            raise ValueError(f"Invalid side '{request.side}'; expected 'buy' or 'sell'")
        if request.order_type not in {"market", "limit", "stop", "bracket"}:
            raise ValueError(f"Invalid order_type '{request.order_type}'")
        if request.quantity <= 0:
            raise ValueError("OrderRequest.quantity must be positive")

        if request.order_type == "limit" and request.limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        if request.order_type == "stop" and request.stop_price is None:
            raise ValueError("stop_price is required for stop orders")
        if request.order_type == "bracket":
            missing = []
            if request.stop_price is None:
                missing.append("stop_price")
            if request.stop_loss is None:
                missing.append("stop_loss")
            if request.take_profit is None:
                missing.append("take_profit")
            if missing:
                raise ValueError(f"Bracket orders require {', '.join(missing)}")

        if request.time_in_force not in {"GTC", "IOC", "FOK"}:
            logger.debug(
                "Uncommon time_in_force '%s' used; proceeding without strict validation",
                request.time_in_force,
            )

    def _validate_symbol(self, symbol: str) -> None:
        """Ensure a symbol string is well‑formed."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non‑empty string")

    # --------------------------------------------------------------------- #
    # Abstract broker methods
    # --------------------------------------------------------------------- #

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
    async def get_positions(self) -> List[dict]:
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
    ) -> List[dict]:
        """Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}."""