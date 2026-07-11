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

    def __post_init__(self) -> None:
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
        if self.limit_price is not None:
            if not isinstance(self.limit_price, (int, float)) or self.limit_price <= 0:
                raise ValueError("limit_price must be a positive number when provided")
        if self.stop_price is not None:
            if not isinstance(self.stop_price, (int, float)) or self.stop_price <= 0:
                raise ValueError("stop_price must be a positive number when provided")
        if self.stop_loss is not None:
            if not isinstance(self.stop_loss, (int, float)) or self.stop_loss <= 0:
                raise ValueError("stop_loss must be a positive number when provided")
        if self.take_profit is not None:
            if not isinstance(self.take_profit, (int, float)) or self.take_profit <= 0:
                raise ValueError("take_profit must be a positive number when provided")
        if not isinstance(self.time_in_force, str) or not self.time_in_force:
            raise ValueError("time_in_force must be a non‑empty string")
        if not isinstance(self.account_id, str):
            raise ValueError("account_id must be a string")
        if not isinstance(self.risk_bucket, str) or not self.risk_bucket:
            raise ValueError("risk_bucket must be a non‑empty string")
        if self.execution_algo not in {"market", "limit_first", "twap", "vwap"}:
            raise ValueError(
                "execution_algo must be one of 'market', 'limit_first', 'twap', or 'vwap'"
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
            raise ValueError("request must be an instance of OrderRequest")
        # Validation of OrderRequest fields occurs in its __post_init__
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