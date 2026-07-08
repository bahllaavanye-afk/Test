from abc import ABC, abstractmethod
from dataclasses import dataclass
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

        if not isinstance(request.symbol, str) or not request.symbol:
            raise ValueError("symbol must be a non‑empty string")

        if request.side not in {"buy", "sell"}:
            raise ValueError("side must be either 'buy' or 'sell'")

        if request.order_type not in {"market", "limit", "stop", "bracket"}:
            raise ValueError(
                "order_type must be one of 'market', 'limit', 'stop', or 'bracket'"
            )

        if not isinstance(request.quantity, (int, float)) or request.quantity <= 0:
            raise ValueError("quantity must be a positive number")

        if request.order_type in {"limit", "bracket"}:
            if request.limit_price is None:
                raise ValueError("limit_price is required for limit or bracket orders")
            if not isinstance(request.limit_price, (int, float)):
                raise ValueError("limit_price must be a number")

        if request.order_type in {"stop", "bracket"}:
            if request.stop_price is None:
                raise ValueError("stop_price is required for stop or bracket orders")
            if not isinstance(request.stop_price, (int, float)):
                raise ValueError("stop_price must be a number")

        if request.order_type == "bracket":
            if request.stop_loss is None or request.take_profit is None:
                raise ValueError(
                    "stop_loss and take_profit are required for bracket orders"
                )
            if not isinstance(request.stop_loss, (int, float)):
                raise ValueError("stop_loss must be a number")
            if not isinstance(request.take_profit, (int, float)):
                raise ValueError("take_profit must be a number")

        if not isinstance(request.time_in_force, str) or not request.time_in_force:
            raise ValueError("time_in_force must be a non‑empty string")

        if not isinstance(request.account_id, str):
            raise ValueError("account_id must be a string")

        if request.strategy_id is not None and not isinstance(request.strategy_id, str):
            raise ValueError("strategy_id must be a string if provided")

        if not isinstance(request.risk_bucket, str) or not request.risk_bucket:
            raise ValueError("risk_bucket must be a non‑empty string")

        if request.execution_algo not in {
            "market",
            "limit_first",
            "twap",
            "vwap",
        }:
            raise ValueError(
                "execution_algo must be one of 'market', 'limit_first', 'twap', or 'vwap'"
            )

        raise NotImplementedError("place_order must be implemented by subclass")

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")
        raise NotImplementedError("cancel_order must be implemented by subclass")

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> dict:
        """Get current status of an order."""
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")
        raise NotImplementedError("get_order must be implemented by subclass")

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return all open positions."""
        raise NotImplementedError("get_positions must be implemented by subclass")

    @abstractmethod
    async def get_account(self) -> dict:
        """Return account balance and equity."""
        raise NotImplementedError("get_account must be implemented by subclass")

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol."""
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")
        raise NotImplementedError("get_quote must be implemented by subclass")

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}."""
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")
        if not isinstance(interval, str) or not interval:
            raise ValueError("interval must be a non‑empty string")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        raise NotImplementedError("get_historical must be implemented by subclass")