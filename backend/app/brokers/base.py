from abc import ABC, abstractmethod
from dataclasses import dataclass
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
        # Input validation
        if not isinstance(request, OrderRequest):
            raise ValueError("place_order expects an OrderRequest instance.")
        if not isinstance(request.symbol, str) or not request.symbol:
            raise ValueError("OrderRequest.symbol must be a non‑empty string.")
        if request.side not in {"buy", "sell"}:
            raise ValueError("OrderRequest.side must be either 'buy' or 'sell'.")
        if request.order_type not in {"market", "limit", "stop", "bracket"}:
            raise ValueError(
                "OrderRequest.order_type must be one of: market, limit, stop, bracket."
            )
        if not isinstance(request.quantity, (int, float)) or request.quantity <= 0:
            raise ValueError("OrderRequest.quantity must be a positive number.")
        if request.order_type == "limit":
            if request.limit_price is None:
                raise ValueError("limit_price is required for limit orders.")
        if request.order_type in {"stop", "bracket"}:
            if request.stop_price is None:
                raise ValueError("stop_price is required for stop or bracket orders.")
        if request.order_type == "bracket":
            if request.stop_loss is None or request.take_profit is None:
                raise ValueError(
                    "Both stop_loss and take_profit are required for bracket orders."
                )
            if request.stop_loss <= 0 or request.take_profit <= 0:
                raise ValueError("stop_loss and take_profit must be positive numbers.")
        if request.time_in_force not in {"GTC", "IOC", "FOK", "DAY"}:
            raise ValueError(
                "time_in_force must be one of: GTC, IOC, FOK, DAY."
            )
        if not isinstance(request.account_id, str):
            raise ValueError("account_id must be a string.")
        if not isinstance(request.execution_algo, str):
            raise ValueError("execution_algo must be a string.")
        # Subclass implementation must be provided
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string.")
        raise NotImplementedError

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> dict:
        """Get current status of an order."""
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string.")
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
            raise ValueError("symbol must be a non‑empty string.")
        raise NotImplementedError

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[dict]:
        """Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}."""
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string.")
        if not isinstance(interval, str) or not interval:
            raise ValueError("interval must be a non‑empty string.")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")
        raise NotImplementedError