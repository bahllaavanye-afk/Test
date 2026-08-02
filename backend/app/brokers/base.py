from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
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

    # Simple in‑memory caches for frequently accessed data
    _historical_cache: Dict[Tuple[str, str, int], List[dict]] = {}
    _quote_cache: Dict[str, Tuple[float, QuoteResult]] = {}
    _quote_ttl: float = 5.0  # seconds

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

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol with short‑term caching."""
        now = time.time()
        cached = self._quote_cache.get(symbol)
        if cached and now - cached[0] < self._quote_ttl:
            return cached[1]
        result = await self._fetch_quote(symbol)
        self._quote_cache[symbol] = (now, result)
        return result

    @abstractmethod
    async def _fetch_quote(self, symbol: str) -> QuoteResult:
        """Broker‑specific implementation to fetch a fresh quote."""

    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """Return OHLCV bars with caching to avoid duplicate remote calls."""
        cache_key = (symbol, interval, limit)
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]
        data = await self._fetch_historical(symbol, interval, limit)
        self._historical_cache[cache_key] = data
        return data

    @abstractmethod
    async def _fetch_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """Broker‑specific implementation to retrieve historical OHLCV data."""