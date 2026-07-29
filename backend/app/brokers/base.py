from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import asyncio
import time


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

    # Internal cache for historical data: {(symbol, interval, limit): (timestamp, data)}
    _historical_cache: Dict[Tuple[str, str, int], Tuple[float, List[dict]]] = {}
    _cache_lock = asyncio.Lock()

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

    async def _get_historical_cached(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        ttl: int = 300,
    ) -> List[dict]:
        """
        Retrieve historical data with in‑memory caching.

        Parameters
        ----------
        symbol: str
            Trading symbol.
        interval: str
            Bar interval (e.g., "1m", "5m").
        limit: int, default 500
            Number of bars to fetch.
        ttl: int, default 300
            Time‑to‑live for cache entries in seconds.

        Returns
        -------
        List[dict]
            List of OHLCV dictionaries.
        """
        if limit <= 0:
            return []

        cache_key = (symbol, interval, limit)
        async with self._cache_lock:
            entry = self._historical_cache.get(cache_key)
            now = time.time()
            if entry:
                timestamp, data = entry
                if now - timestamp < ttl:
                    return data

        # Fetch fresh data from the concrete implementation
        data = await self.get_historical(symbol, interval, limit)

        async with self._cache_lock:
            self._historical_cache[cache_key] = (time.time(), data)

        return data