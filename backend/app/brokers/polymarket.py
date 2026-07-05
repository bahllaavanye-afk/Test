"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
Optimized with simple in-memory caching for network‑heavy calls.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Tuple

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

    _MARKET_CACHE_TTL = 60  # seconds
    _ORDER_BOOK_CACHE_TTL = 30  # seconds

    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )
        # Simple caches: {cache_key: (timestamp, data)}
        self._market_cache: Tuple[float, List[Dict[str, Any]]] = (0.0, [])
        self._order_book_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    async def _fetch_markets(self) -> List[Dict[str, Any]]:
        """Fetch raw market list from Polymarket, bypassing cache."""
        return await asyncio.to_thread(self.client.get_markets)

    async def get_markets(self, min_open_interest: float = 10000) -> List[Dict[str, Any]]:
        """
        Auto‑discover active markets with sufficient liquidity.

        Results are cached for ``_MARKET_CACHE_TTL`` seconds to reduce
        repeated network calls.
        """
        now = time.time()
        cache_timestamp, cached_data = self._market_cache
        if now - cache_timestamp < self._MARKET_CACHE_TTL:
            markets = cached_data
        else:
            try:
                markets = await self._fetch_markets()
                self._market_cache = (now, markets)
            except Exception as e:  # pragma: no cover – defensive
                logger.error("Polymarket market fetch failed", error=str(e))
                return []

        if min_open_interest <= 0:
            return markets
        return [
            m
            for m in markets
            if float(m.get("openInterest", 0)) >= min_open_interest
        ]

    async def _fetch_order_book(self, token_id: str) -> Dict[str, Any]:
        """Fetch raw order book for a token, bypassing cache."""
        return await asyncio.to_thread(self.client.get_order_book, token_id)

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Retrieve order book for ``token_id`` with caching.

        Cache is scoped per token and expires after ``_ORDER_BOOK_CACHE_TTL`` seconds.
        """
        now = time.time()
        cached = self._order_book_cache.get(token_id)
        if cached:
            ts, data = cached
            if now - ts < self._ORDER_BOOK_CACHE_TTL:
                return data

        try:
            ob = await self._fetch_order_book(token_id)
            self._order_book_cache[token_id] = (now, ob)
            return ob
        except Exception as e:  # pragma: no cover – defensive
            logger.error("Polymarket order book fetch failed", token_id=token_id, error=str(e))
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
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Retrieve a specific order's details."""
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Polymarket does not expose position data via this API."""
        return []

    async def get_account(self) -> Dict[str, Any]:
        """Polymarket does not expose account balances via this API."""
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Calculate best bid/ask and mid price from the order book."""
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        return QuoteResult(
            symbol=symbol,
            bid=best_bid,
            ask=best_ask,
            last=(best_bid + best_ask) / 2,
        )

    async def get_historical(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Polymarket doesn't have traditional OHLCV data."""
        return []