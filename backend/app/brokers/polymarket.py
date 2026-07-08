"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
Optimized with lightweight in‑memory caching for frequently accessed data.
"""
from datetime import datetime, timedelta
from typing import List, Dict

import asyncio

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


class PolymarketBroker(AbstractBroker):
    """Broker implementation for Polymarket's CLOB."""

    _MARKETS_CACHE_TTL = timedelta(minutes=5)
    _ORDER_BOOK_CACHE_TTL = timedelta(seconds=5)

    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )
        self._markets_cache: Dict[str, datetime] = {}
        self._order_book_cache: Dict[str, tuple[datetime, dict]] = {}

    async def get_markets(self, min_open_interest: float = 10000) -> List[dict]:
        """Return active markets filtered by open interest, using a short‑lived cache."""
        cache_key = f"markets_{min_open_interest}"
        now = datetime.utcnow()

        # Return cached result if still fresh
        if cache_key in self._markets_cache:
            cached_time, cached_data = self._markets_cache[cache_key]
            if now - cached_time < self._MARKETS_CACHE_TTL:
                return cached_data

        try:
            markets = await asyncio.to_thread(self.client.get_markets)
            filtered = [
                m for m in markets
                if float(m.get("openInterest", 0)) >= min_open_interest
            ]
            # Store in cache
            self._markets_cache[cache_key] = (now, filtered)
            return filtered
        except Exception as e:
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> dict:
        """Fetch the order book for a token, memoizing recent calls."""
        now = datetime.utcnow()
        cached = self._order_book_cache.get(token_id)

        if cached:
            cached_time, cached_data = cached
            if now - cached_time < self._ORDER_BOOK_CACHE_TTL:
                return cached_data

        try:
            ob = await asyncio.to_thread(self.client.get_order_book, token_id)
            self._order_book_cache[token_id] = (now, ob)
            return ob
        except Exception as e:
            logger.error(
                "Polymarket order book fetch failed",
                token_id=token_id,
                error=str(e)
            )
            return {"bids": [], "asks": []}

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Create and post a new order on Polymarket."""
        try:
            args = OrderArgs(
                token_id=request.symbol,
                price=request.limit_price or 0.5,
                size=request.quantity,
                side=request.side.upper(),
            )
            order = await asyncio.to_thread(self.client.create_and_post_order, args)
            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                raw_payload=order,
            )
        except Exception as e:
            raise BrokerError(f"Polymarket: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except Exception as e:
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e)
            )
            return False

    async def get_order(self, broker_order_id: str) -> dict:
        """Retrieve details of a specific order."""
        try:
            return await asyncio.to_thread(self.client.get_order, broker_order_id)
        except Exception as e:
            logger.error(
                "Polymarket get_order failed",
                order_id=broker_order_id,
                error=str(e)
            )
            return {}

    async def get_positions(self) -> List[dict]:
        """Polymarket does not expose positions via this client."""
        return []

    async def get_account(self) -> dict:
        """Polymarket does not expose account details via this client."""
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Derive a simple quote from the best bid/ask."""
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        mid = (best_bid + best_ask) / 2
        return QuoteResult(symbol=symbol, bid=best_bid, ask=best_ask, last=mid)

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> List[dict]:
        """Polymarket does not provide traditional OHLCV data."""
        return []