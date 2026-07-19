from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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

    def __post_init__(self) -> None:
        # Basic string validations
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non‑empty string")
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be either 'buy' or 'sell'")
        if self.order_type not in {"market", "limit", "stop", "bracket"}:
            raise ValueError(
                "order_type must be one of 'market', 'limit', 'stop', or 'bracket'"
            )
        if not isinstance(self.quantity, (int, float)) or self.quantity <= 0:
            raise ValueError("quantity must be a positive number")

        # Time in force validation
        if self.time_in_force not in {"GTC", "IOC", "FOK", "DAY"}:
            raise ValueError(
                "time_in_force must be one of 'GTC', 'IOC', 'FOK', or 'DAY'"
            )

        # Execution algorithm validation
        if self.execution_algo not in {
            "limit_first",
            "market",
            "twap",
            "vwap",
        }:
            raise ValueError(
                "execution_algo must be one of 'limit_first', 'market', 'twap', or 'vwap'"
            )

        # Conditional price checks
        if self.order_type == "limit":
            if self.limit_price is None:
                raise ValueError("limit_price is required for limit orders")
        if self.order_type == "stop":
            if self.stop_price is None:
                raise ValueError("stop_price is required for stop orders")
        if self.order_type == "bracket":
            if (
                self.stop_loss is None
                or self.take_profit is None
                or self.limit_price is None
                or self.stop_price is None
            ):
                raise ValueError(
                    "limit_price, stop_price, stop_loss, and take_profit are required for bracket orders"
                )

        # Account ID validation (allow empty for internal routing)
        if not isinstance(self.account_id, str):
            raise ValueError("account_id must be a string")

        # Strategy ID validation
        if self.strategy_id is not None and not isinstance(self.strategy_id, str):
            raise ValueError("strategy_id must be a string or None")

        # Risk bucket validation
        if not isinstance(self.risk_bucket, str) or not self.risk_bucket:
            raise ValueError("risk_bucket must be a non‑empty string")


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
            ValueError: If the request is not a valid OrderRequest.
            BrokerError: On failure to place the order (implemented by subclass).
        """
        if not isinstance(request, OrderRequest):
            raise ValueError("request must be an instance of OrderRequest")
        # Validation is performed in OrderRequest.__post_init__
        # Subclass implementations should call super().place_order(request) if they wish
        # to retain this validation step.

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an open order.

        Returns:
            bool: True if cancelled.

        Raises:
            ValueError: If broker_order_id is not a non‑empty string.
        """
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> dict:
        """
        Get current status of an order.

        Raises:
            ValueError: If broker_order_id is not a non‑empty string.
        """
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return all open positions."""
        # No input parameters to validate.

    @abstractmethod
    async def get_account(self) -> dict:
        """Return account balance and equity."""
        # No input parameters to validate.

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """
        Return current bid/ask/last for a symbol.

        Raises:
            ValueError: If symbol is not a non‑empty string.
        """
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """
        Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")
        if not isinstance(interval, str) or not interval:
            raise ValueError("interval must be a non‑empty string")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")