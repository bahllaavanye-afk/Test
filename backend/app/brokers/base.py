from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List


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

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("OrderRequest.symbol must be a non‑empty string")
        if self.side not in {"buy", "sell"}:
            raise ValueError("OrderRequest.side must be 'buy' or 'sell'")
        if self.order_type not in {"market", "limit", "stop", "bracket"}:
            raise ValueError(
                "OrderRequest.order_type must be one of 'market', 'limit', 'stop', 'bracket'"
            )
        if not isinstance(self.quantity, (int, float)) or self.quantity <= 0:
            raise ValueError("OrderRequest.quantity must be a positive number")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price must be provided for limit orders")
        if self.order_type == "stop" and self.stop_price is None:
            raise ValueError("stop_price must be provided for stop orders")
        if self.order_type == "bracket":
            if self.stop_loss is None or self.take_profit is None:
                raise ValueError(
                    "stop_loss and take_profit must be provided for bracket orders"
                )
        if not isinstance(self.time_in_force, str) or not self.time_in_force:
            raise ValueError("OrderRequest.time_in_force must be a non‑empty string")
        if self.execution_algo not in {"market", "limit_first", "twap", "vwap"}:
            raise ValueError(
                "OrderRequest.execution_algo must be one of 'market', 'limit_first', 'twap', 'vwap'"
            )


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
        if not isinstance(request, OrderRequest):
            raise ValueError("place_order expects an OrderRequest instance")
        # Validation is performed in OrderRequest.__post_init__
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")
        raise NotImplementedError

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> dict:
        """Get current status of an order."""
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")
        raise NotImplementedError

    @abstractmethod
    async def get_positions(self) -> List[dict]:
        """Return all open positions."""
        raise NotImplementedError

    @abstractmethod
    async def get_account(self) -> dict:
        """Return account balance and equity."""
        raise NotImplementedError

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol."""
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")
        raise NotImplementedError

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[dict]:
        """Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}."""
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")
        if not isinstance(interval, str) or not interval:
            raise ValueError("interval must be a non‑empty string")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        raise NotImplementedError