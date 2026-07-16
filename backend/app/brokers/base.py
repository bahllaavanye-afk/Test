from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Dict


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
                "OrderRequest.order_type must be one of: market, limit, stop, bracket"
            )
        if not isinstance(self.quantity, (int, float)) or self.quantity <= 0:
            raise ValueError("OrderRequest.quantity must be a positive number")
        if self.limit_price is not None and (
            not isinstance(self.limit_price, (int, float)) or self.limit_price <= 0
        ):
            raise ValueError("OrderRequest.limit_price must be a positive number if provided")
        if self.stop_price is not None and (
            not isinstance(self.stop_price, (int, float)) or self.stop_price <= 0
        ):
            raise ValueError("OrderRequest.stop_price must be a positive number if provided")
        if self.stop_loss is not None and (
            not isinstance(self.stop_loss, (int, float)) or self.stop_loss <= 0
        ):
            raise ValueError("OrderRequest.stop_loss must be a positive number if provided")
        if self.take_profit is not None and (
            not isinstance(self.take_profit, (int, float)) or self.take_profit <= 0
        ):
            raise ValueError("OrderRequest.take_profit must be a positive number if provided")
        if not isinstance(self.time_in_force, str) or not self.time_in_force:
            raise ValueError("OrderRequest.time_in_force must be a non‑empty string")
        if not isinstance(self.account_id, str):
            raise ValueError("OrderRequest.account_id must be a string")
        if self.strategy_id is not None and not isinstance(self.strategy_id, str):
            raise ValueError("OrderRequest.strategy_id must be a string or None")
        if not isinstance(self.risk_bucket, str) or not self.risk_bucket:
            raise ValueError("OrderRequest.risk_bucket must be a non‑empty string")
        if self.execution_algo not in {"market", "limit_first", "twap", "vwap"}:
            raise ValueError(
                "OrderRequest.execution_algo must be one of: market, limit_first, twap, vwap"
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
            raise ValueError("place_order expects a valid OrderRequest instance")
        # Sub‑class must implement actual order placement
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("cancel_order requires a non‑empty broker_order_id string")
        raise NotImplementedError

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> dict:
        """Get current status of an order."""
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("get_order requires a non‑empty broker_order_id string")
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
            raise ValueError("get_quote requires a non‑empty symbol string")
        raise NotImplementedError

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[dict]:
        """Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}."""
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("get_historical requires a non‑empty symbol string")
        if not isinstance(interval, str) or not interval:
            raise ValueError("get_historical requires a non‑empty interval string")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("get_historical limit must be a positive integer")
        raise NotImplementedError