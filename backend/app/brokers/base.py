from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional
import asyncio


@dataclass(slots=True)
class OrderRequest:
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


@dataclass(slots=True)
class OrderResult:
    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    raw_payload: Optional[dict] = None


@dataclass(slots=True)
class QuoteResult:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: Optional[float] = None


class AbstractBroker(ABC):
    """Interface that all brokers must implement."""

    # Simple in‑memory cache for historical data
    _historical_cache: Dict[Tuple[str, str, int], List[Dict]] = {}
    _cache_lock: asyncio.Lock = asyncio.Lock()

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
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}."""

    async def get_historical_cached(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """
        Cached wrapper around ``get_historical`` to avoid repeated expensive calls.
        The cache is per‑process and thread‑safe via an asyncio lock.
        """
        key = (symbol, interval, limit)
        async with self._cache_lock:
            cached = self._historical_cache.get(key)
            if cached is not None:
                return cached

        data = await self.get_historical(symbol, interval, limit)

        async with self._cache_lock:
            # Store result for future calls
            self._historical_cache[key] = data

        return data