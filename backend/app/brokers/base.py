from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


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
    """Interface that all brokers must implement.

    Provides a cached implementation for historical data retrieval,
    which is typically the most expensive operation.
    """

    # Simple in‑memory cache for historical OHLCV data.
    # Key: (symbol, interval, limit) -> List[dict]
    _historical_cache: Dict[Tuple[str, str, int], List[dict]] = {}

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
    async def _fetch_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[dict]:
        """Concrete brokers must implement the raw historical data fetch."""

    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[dict]:
        """Return OHLCV bars with in‑memory caching.

        The cache key includes symbol, interval, and limit. If the same
        request is made again, the cached result is returned immediately,
        avoiding the expensive network call.
        """
        if limit <= 0:
            return []

        key = (symbol, interval, limit)
        cached = self._historical_cache.get(key)
        if cached is not None:
            return cached

        data = await self._fetch_historical(symbol, interval, limit)
        # Store a shallow copy to prevent accidental mutation of cached data.
        self._historical_cache[key] = list(data)
        return data