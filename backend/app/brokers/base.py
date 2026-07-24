from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import asyncio
from collections import OrderedDict


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

    Provides cached implementations for quote and historical data retrieval,
    which are typically the most expensive operations.
    """

    # Simple in‑memory caches with size limits
    _quote_cache: Dict[str, QuoteResult] = {}
    _historical_cache: OrderedDict[Tuple[str, str, int], List[dict]] = OrderedDict()
    _cache_lock = asyncio.Lock()
    _MAX_HISTORICAL_CACHE_SIZE = 128

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
    async def _fetch_quote(self, symbol: str) -> QuoteResult:
        """Low‑level broker call to retrieve a fresh quote."""

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return a cached quote if available; otherwise fetch a fresh one."""
        async with self._cache_lock:
            cached = self._quote_cache.get(symbol)
            if cached is not None:
                return cached

        fresh = await self._fetch_quote(symbol)

        async with self._cache_lock:
            self._quote_cache[symbol] = fresh
        return fresh

    @abstractmethod
    async def _fetch_historical(
        self, symbol: str, interval: str, limit: int
    ) -> List[dict]:
        """Low‑level broker call to retrieve historical OHLCV data."""

    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[dict]:
        """Return cached historical data if present; otherwise fetch and cache."""
        key = (symbol, interval, limit)

        async with self._cache_lock:
            if key in self._historical_cache:
                # Move accessed item to the end to reflect recent use
                self._historical_cache.move_to_end(key)
                return self._historical_cache[key]

        data = await self._fetch_historical(symbol, interval, limit)

        async with self._cache_lock:
            # Evict oldest entry if cache exceeds size limit
            if len(self._historical_cache) >= self._MAX_HISTORICAL_CACHE_SIZE:
                self._historical_cache.popitem(last=False)
            self._historical_cache[key] = data
        return data