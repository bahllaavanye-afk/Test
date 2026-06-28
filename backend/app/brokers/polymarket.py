"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
"""
from __future__ import annotations

import json as _json
import time
import urllib.request as _urllib_request
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    POLY_AVAILABLE = True
except ImportError:
    POLY_AVAILABLE = False


class PolymarketBroker(AbstractBroker):
    """Broker implementation for Polymarket CLOB."""

    _MARKET_CACHE_TTL = 60.0  # seconds

    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )
        self._market_cache: Tuple[float, List[Dict[str, Any]]] = (0.0, [])

    async def get_markets(self, min_open_interest: float = 10000) -> List[Dict[str, Any]]:
        """Auto-discover active markets with sufficient liquidity.

        Results are cached for a short period to avoid repeated network calls.
        """
        now = time.time()
        cache_timestamp, cached_markets = self._market_cache
        if now - cache_timestamp < self._MARKET_CACHE_TTL:
            markets = cached_markets
        else:
            try:
                import asyncio
                markets = await asyncio.to_thread(self.client.get_markets)
                self._market_cache = (now, markets)
            except Exception as e:
                logger.error("Polymarket market fetch failed", error=str(e))
                markets = []

        # Filter by open interest; this operation is cheap compared to the fetch.
        return [m for m in markets if float(m.get("openInterest", 0)) >= min_open_interest]

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        import asyncio
        return await asyncio.to_thread(self.client.get_order_book, token_id)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        try:
            import asyncio
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
        try:
            import asyncio
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except Exception:
            return False

    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        import asyncio
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> List[Dict[str, Any]]:
        return []

    async def get_account(self) -> Dict[str, Any]:
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        return QuoteResult(symbol=symbol, bid=best_bid, ask=best_ask, last=(best_bid + best_ask) / 2)

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> List[Dict[str, Any]]:
        return []  # Polymarket doesn't have traditional OHLCV


class PolymarketPublicClient:
    """Read‑only Polymarket CLOB client — no API key required."""
    BASE = "https://clob.polymarket.com"

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | List[Any]:
        url = f"{self.BASE}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        try:
            with _urllib_request.urlopen(url, timeout=10) as resp:
                return _json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("PolymarketPublicClient fetch failed", url=url, error=str(exc))
            return {}

    @lru_cache(maxsize=32)
    def get_markets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch active markets sorted by volume.

        Results are cached via LRU to reduce repeated HTTP calls.
        """
        data = self._get("/markets", {"limit": str(limit), "active": "true"})
        if isinstance(data, dict):
            return data.get("data", []) or []
        if isinstance(data, list):
            return data
        return []

    @lru_cache(maxsize=128)
    def get_last_price(self, token_id: str) -> float | None:
        """Return last YES token price (0–1 range)."""
        data = self._get("/last-trade-price", {"token_id": token_id})
        if isinstance(data, dict):
            price = data.get("price")
            if price is not None:
                try:
                    return float(price)
                except (TypeError, ValueError):
                    pass
        return None