"""Polymarket read-only market data endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/polymarket", tags=["polymarket"])


def _client():
    from app.brokers.polymarket import PolymarketPublicClient
    return PolymarketPublicClient()


@router.get("/markets")
async def list_markets(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    """List active Polymarket prediction markets."""
    loop = asyncio.get_running_loop()
    try:
        markets = await loop.run_in_executor(None, lambda: _client().get_markets(limit))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Polymarket unavailable: {exc}")
    return {"markets": markets, "count": len(markets)}


@router.get("/prices")
async def get_prices(
    current_user: User = Depends(get_current_user),
):
    """Return current prices for all tracked Polymarket markets."""
    from app.redis_client import price_cache
    loop = asyncio.get_running_loop()
    try:
        markets = await loop.run_in_executor(None, lambda: _client().get_markets(20))
    except Exception:
        return {"prices": []}
    prices = []
    for m in markets:
        token_id = (m.get("tokens") or [{}])[0].get("token_id", "")
        if not token_id:
            continue
        cached = await price_cache.get_price("polymarket", token_id)
        prices.append({
            "condition_id": m.get("condition_id"),
            "question": m.get("question", ""),
            "token_id": token_id,
            "price": cached.get("last") if cached else None,
        })
    return {"prices": prices}
