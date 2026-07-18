from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class OrderRequest:
    """Data container for an order submission."""

    symbol: str
    side: str               # buy|sell
    order_type: str         # market|limit|stop|bracket
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None      # for bracket orders
    take_profit: Optional[float] = None    # for bracket orders
    time_in_force: str = "GTC"
    account_id: str = ""
    strategy_id: Optional[str] = None
    risk_bucket: str = "directional"   # for risk manager routing
    execution_algo: str = "limit_first"  # market|limit_first|twap|vwap

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of the request."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OrderRequest":
        """Create an OrderRequest instance from a dictionary."""
        return OrderRequest(**data)


@dataclass(slots=True)
class OrderResult:
    """Result returned after an order is placed."""

    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    raw_payload: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of the result."""
        return asdict(self)


@dataclass(slots=True)
class QuoteResult:
    """Quote information for a given symbol."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of the quote."""
        return asdict(self)


class AbstractBroker(ABC):
    """Interface that all broker implementations must follow."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker. Raises BrokerError on failure."""

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict:
        """Get current status of an order."""

    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Return all open positions."""

    @abstractmethod
    async def get_account(self) -> Dict:
        """Return account balance and equity."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol."""

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict]:
        """Return OHLCV bars.
        Each dict contains: {ts, open, high, low, close, volume}.
        """