"""Kalshi read-only market data endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/kalshi", tags=["kalshi"])


def _client():
    from app.brokers.kalshi import KalshiPublicClient
    return KalshiPublicClient()


@router.get("/events")
async def list_events(
    limit: int = 25,
    current_user: User = Depends(get_current_user),
):
    """List open Kalshi prediction events."""
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")
    loop = asyncio.get_running_loop()
    try:
        events = await loop.run_in_executor(None, lambda: _client().get_events(limit=limit))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Kalshi unavailable: {exc}")
    return {"events": events, "count": len(events)}


@router.get("/markets/{ticker}")
async def get_market(
    ticker: str,
    current_user: User = Depends(get_current_user),
):
    """Return single Kalshi market detail."""
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    loop = asyncio.get_running_loop()
    try:
        market = await loop.run_in_executor(None, lambda: _client().get_market(ticker))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Kalshi unavailable: {exc}")
    if not market:
        raise HTTPException(status_code=404, detail="Market not found")
    return market