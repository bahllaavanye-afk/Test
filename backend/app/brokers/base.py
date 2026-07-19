from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple
from time import monotonic


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

    Provides default caching for quote and historical data to avoid
    redundant network calls, which are typically the most expensive
    operations in broker interactions.
    """

    # Simple in‑memory caches: {(symbol, interval, limit): (timestamp, data)}
    _quote_cache: Dict[str, Tuple[float, QuoteResult]] = {}
    _historical_cache: Dict[Tuple[str, str, int], Tuple[float, List[Dict]]] = {}

    # --------------------------------------------------------------------- #
    # Core abstract methods that concrete brokers must implement
    # --------------------------------------------------------------------- #
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
    async def _fetch_quote(self, symbol: str) -> QuoteResult:
        """Low‑level fetch of quote data from the broker."""

    @abstractmethod
    async def _fetch_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[dict]:
        """Low‑level fetch of OHLCV bars from the broker."""

    # --------------------------------------------------------------------- #
    # Cached public helpers
    # --------------------------------------------------------------------- #
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol with short‑term caching.

        Cache duration is 1 second to prevent burst requests for the same
        symbol within a tight loop.
        """
        now = monotonic()
        cached = self._quote_cache.get(symbol)
        if cached and now - cached[0] < 1.0:
            return cached[1]

        quote = await self._fetch_quote(symbol)
        self._quote_cache[symbol] = (now, quote)
        return quote

    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[dict]:
        """Return OHLCV bars with a 5‑minute cache to reduce expensive calls.

        The cache key includes symbol, interval, and limit because each
        combination may produce a distinct dataset.
        """
        cache_key = (symbol, interval, limit)
        now = monotonic()
        cached = self._historical_cache.get(cache_key)
        if cached and now - cached[0] < 300.0:  # 5 minutes
            return cached[1]

        data = await self._fetch_historical(symbol, interval, limit)
        self._historical_cache[cache_key] = (now, data)
        return data