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
    """Interface that all brokers must implement."""

    # Simple in‑memory cache for historical data.
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

    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """
        Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}.

        This implementation adds a lightweight in‑memory cache to avoid
        repeated expensive calls for identical requests during the same
        process lifetime.
        """
        cache_key = (symbol, interval, limit)
        cached = self._historical_cache.get(cache_key)
        if cached is not None:
            return cached

        data = await self._fetch_historical(symbol, interval, limit)
        # Store the result for future identical requests.
        self._historical_cache[cache_key] = data
        return data

    @abstractmethod
    async def _fetch_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """
        Concrete brokers must implement the actual retrieval of historical data.
        The default ``get_historical`` method will call this and cache the result.
        """