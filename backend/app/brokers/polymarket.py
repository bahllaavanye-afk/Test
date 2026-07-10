"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
"""
import asyncio
import time
from typing import Any, Dict, List

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger
from app.config import settings

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    POLY_AVAILABLE = True
except ImportError:
    POLY_AVAILABLE = False


# Simple in‑memory async cache with TTL
class AsyncCache:
    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self._store: Dict[str, Any] = {}
        self._expirations: Dict[str, float] = {}

    async def get(self, key: str, loader):
        now = time.time()
        if key in self._store and now < self._expirations.get(key, 0):
            return self._store[key]
        value = await loader()
        self._store[key] = value
        self._expirations[key] = now + self.ttl
        return value

    def invalidate(self, key: str):
        self._store.pop(key, None)
        self._expirations.pop(key, None)


class PolymarketBroker(AbstractBroker):
    _markets_cache = AsyncCache(ttl=300)          # cache market list for 5 minutes
    _order_book_cache = AsyncCache(ttl=10)        # cache order book for 10 seconds

    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    async def get_markets(self, min_open_interest: float = 10000) -> List[Dict]:
        """Auto‑discover active markets with sufficient liquidity, cached for performance."""
        async def loader():
            try:
                markets = await asyncio.to_thread(self.client.get_markets)
                return [m for m in markets if float(m.get("openInterest", 0)) >= min_open_interest]
            except Exception as e:
                logger.error("Polymarket market fetch failed", error=str(e))
                return []

        return await self._markets_cache.get(f"markets:{min_open_interest}", loader)

    async def get_order_book(self, token_id: str) -> Dict:
        """Retrieve order book for a token, cached briefly to reduce repeated calls."""
        async def loader():
            return await asyncio.to_thread(self.client.get_order_book, token_id)

        return await self._order_book_cache.get(f"ob:{token_id}", loader)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Create and post an order; invalidates related caches on success."""
        try:
            args = OrderArgs(
                token_id=request.symbol,
                price=request.limit_price or 0.5,
                size=request.quantity,
                side=request.side.upper(),
            )
            order = await asyncio.to_thread(self.client.create_and_post_order, args)
            # Invalidate caches that may be affected by a new order
            self._order_book_cache.invalidate(f"ob:{request.symbol}")
            self._markets_cache.invalidate("markets")
            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                raw_payload=order,
            )
        except Exception as e:
            raise BrokerError(f"Polymarket: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order and clear relevant caches."""
        try:
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            # Order cancellation may affect order book; conservative cache clear
            self._order_book_cache.invalidate(f"ob:{broker_order_id}")
            return True
        except Exception as e:
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str) -> Dict:
        """Fetch a single order by its broker ID."""
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> List[Dict]:
        """Polymarket does not expose positions via the CLOB client."""
        return []

    async def get_account(self) -> Dict:
        """Polymarket does not expose account details via the CLOB client."""
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return best bid/ask and mid price; leverages cached order book."""
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        mid_price = (best_bid + best_ask) / 2 if (best_bid or best_ask) else 0.0
        return QuoteResult(symbol=symbol, bid=best_bid, ask=best_ask, last=mid_price)

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> List[Dict]:
        """Polymarket doesn't provide traditional OHLCV data."""
        return []