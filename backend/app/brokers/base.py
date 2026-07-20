from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional
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

    def __init__(self) -> None:
        # Simple in‑memory caches for expensive calls.
        # Keys are tuples of request parameters.
        self._historical_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self._quote_cache: Dict[str, Dict[str, Any]] = {}
        self._historical_lock = asyncio.Lock()
        self._quote_lock = asyncio.Lock()

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

    # -------------------------------------------------------------------------
    # Optimized helpers with caching
    # -------------------------------------------------------------------------

    async def get_historical_cached(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        ttl: int = 60,
    ) -> List[dict]:
        """
        Cached version of ``get_historical``.
        Results are kept for ``ttl`` seconds (default 60) to avoid repeated
        expensive network calls. The underlying concrete broker must implement
        ``get_historical``; this wrapper adds only caching logic.
        """
        key = (symbol, interval, limit)
        now = time.time()

        async with self._historical_lock:
            entry = self._historical_cache.get(key)
            if entry and (now - entry["ts"] < ttl):
                return entry["data"]

        # Cache miss – fetch fresh data
        data = await self.get_historical(symbol, interval, limit)

        async with self._historical_lock:
            self._historical_cache[key] = {"ts": time.time(), "data": data}
        return data

    async def get_quote_cached(
        self,
        symbol: str,
        ttl: int = 5,
    ) -> QuoteResult:
        """
        Cached version of ``get_quote``.
        Quote data is typically volatile; a short TTL (default 5 seconds) balances
        freshness with reduced request frequency.
        """
        now = time.time()

        async with self._quote_lock:
            entry = self._quote_cache.get(symbol)
            if entry and (now - entry["ts"] < ttl):
                return entry["data"]

        # Cache miss – fetch fresh quote
        quote = await self.get_quote(symbol)

        async with self._quote_lock:
            self._quote_cache[symbol] = {"ts": time.time(), "data": quote}
        return quote