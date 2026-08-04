from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


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


def _filter_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *data* with keys whose values are ``None`` removed."""
    return {k: v for k, v in data.items() if v is not None}


def order_request_to_dict(request: OrderRequest) -> Dict[str, Any]:
    """
    Convert an :class:`OrderRequest` into a plain ``dict`` suitable for
    serialisation or broker API payloads.

    Fields with a value of ``None`` are omitted to keep the payload compact.
    """
    return _filter_none(asdict(request))


def order_result_from_dict(data: Dict[str, Any]) -> OrderResult:
    """
    Create an :class:`OrderResult` from a mapping. Missing optional fields are
    defaulted according to the dataclass definition.
    """
    return OrderResult(**data)


def quote_result_from_dict(data: Dict[str, Any]) -> QuoteResult:
    """
    Create a :class:`QuoteResult` from a mapping.
    """
    return QuoteResult(**data)


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