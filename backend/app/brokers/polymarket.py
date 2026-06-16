"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
"""
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
    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    async def get_markets(self, min_open_interest: float = 10000) -> list[dict]:
        """Auto-discover active markets with sufficient liquidity."""
        try:
            import asyncio
            markets = await asyncio.to_thread(self.client.get_markets)
            return [m for m in markets if float(m.get("openInterest", 0)) >= min_open_interest]
        except Exception as e:
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> dict:
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

    async def get_order(self, broker_order_id: str) -> dict:
        import asyncio
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> list[dict]:
        return []

    async def get_account(self) -> dict:
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        return QuoteResult(symbol=symbol, bid=best_bid, ask=best_ask, last=(best_bid + best_ask) / 2)

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[dict]:
        return []  # Polymarket doesn't have traditional OHLCV


import json as _json
import urllib.request as _urllib_request


class PolymarketPublicClient:
    """Read-only Polymarket CLOB client — no API key required."""
    BASE = "https://clob.polymarket.com"

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{self.BASE}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        try:
            with _urllib_request.urlopen(url, timeout=10) as resp:
                return _json.loads(resp.read().decode())
        except Exception as exc:
            from app.utils.logging import logger
            logger.debug("PolymarketPublicClient fetch failed", url=url, error=str(exc))
            return {}

    def get_markets(self, limit: int = 50) -> list[dict]:
        """Fetch active markets sorted by volume."""
        data = self._get("/markets", {"limit": str(limit), "active": "true"})
        if isinstance(data, dict):
            return data.get("data", []) or []
        if isinstance(data, list):
            return data
        return []

    def get_last_price(self, token_id: str) -> float | None:
        """Return last YES token price (0–1 range)."""
        data = self._get("/last-trade-price", {"token_id": token_id})
        if isinstance(data, dict):
            price = data.get("price")
            if price is not None:
                return float(price)
        return None
