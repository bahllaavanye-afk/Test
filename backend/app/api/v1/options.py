"""Options trading endpoints: chain, snapshots, expirations, rules validation.

Proxies Alpaca's options API. Uses settings.alpaca_api_key and
settings.alpaca_secret_key directly for market data (no per-account
credentials needed).
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional, Dict, Tuple, List

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_current_user
from app.config import settings
from app.database import get_db
from app.models.user import User
from app.models.account import Account

router = APIRouter(prefix="/options", tags=["options"])

_ALPACA_BASE = "https://paper-api.alpaca.markets"

# Simple in‑memory cache for option snapshots
_SNAPSHOT_CACHE_TTL = 30.0  # seconds
_snapshot_cache: Dict[str, Tuple[float, dict]] = {}


def _alpaca_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        "accept": "application/json",
    }


class EnrichedOptionContract(BaseModel):
    """Fully enriched option contract returned by the API."""

    symbol: str = Field(..., description="Option ticker symbol", example="AAPL240121C00150000")
    underlying_symbol: str = Field(..., description="Underlying equity ticker", example="AAPL")
    expiration_date: date = Field(..., description="Expiration date (YYYY‑MM‑DD)", example="2024-01-21")
    strike_price: float = Field(..., description="Strike price of the option", example=150.0)
    option_type: Literal["call", "put"] = Field(..., description="Option type", example="call")
    bid: Optional[float] = Field(None, description="Best bid price", example=2.35)
    ask: Optional[float] = Field(None, description="Best ask price", example=2.45)
    mid: Optional[float] = Field(None, description="Mid price between bid and ask", example=2.40)
    last: Optional[float] = Field(None, description="Last traded price", example=2.42)
    volume: Optional[int] = Field(None, description="Trading volume for the latest period", example=120)
    open_interest: Optional[int] = Field(None, description="Open interest count", example=450)
    implied_volatility: Optional[float] = Field(
        None, description="Implied volatility (decimal form)", example=0.25
    )
    delta: Optional[float] = Field(None, description="Delta greek", example=0.55)
    gamma: Optional[float] = Field(None, description="Gamma greek", example=0.02)
    theta: Optional[float] = Field(None, description="Theta greek", example=-0.01)
    vega: Optional[float] = Field(None, description="Vega greek", example=0.15)
    rho: Optional[float] = Field(None, description="Rho greek", example=0.05)

    @validator("strike_price")
    def strike_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("strike_price must be positive")
        return v

    @validator("expiration_date", pre=True)
    def parse_expiration(cls, v):
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v


def _enrich_contract(contract: dict, snapshot: dict | None) -> dict:
    """Merge a contract record with its snapshot (Greeks, quotes)."""
    greeks = {}
    iv = None
    bid = None
    ask = None
    mid = None
    volume = None
    last = None

    if snapshot:
        greeks = snapshot.get("greeks") or {}
        iv = snapshot.get("impliedVolatility")
        lq = snapshot.get("latestQuote") or {}
        bid = lq.get("bp")
        ask = lq.get("ap")
        if bid is not None and ask is not None:
            mid = round((bid + ask) / 2, 4)
        lt = snapshot.get("latestTrade") or {}
        last = lt.get("p")
        volume = lt.get("s")

    return {
        "symbol": contract.get("symbol"),
        "underlying_symbol": contract.get("underlying_symbol")
        or contract.get("root_symbol"),
        "expiration_date": contract.get("expiration_date"),
        "strike_price": contract.get("strike_price"),
        "option_type": contract.get("type"),  # "call" | "put"
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": last,
        "volume": volume,
        "open_interest": contract.get("open_interest"),
        "implied_volatility": iv,
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "theta": greeks.get("theta"),
        "vega": greeks.get("vega"),
        "rho": greeks.get("rho"),
    }


def _cache_get(symbol: str) -> dict | None:
    entry = _snapshot_cache.get(symbol)
    if entry:
        ts, data = entry
        if time.time() - ts < _SNAPSHOT_CACHE_TTL:
            return data
        # expired
        del _snapshot_cache[symbol]
    return None


def _cache_set(symbol: str, data: dict) -> None:
    _snapshot_cache[symbol] = (time.time(), data)


async def _fetch_snapshots(symbols: list[str]) -> dict[str, dict]:
    """Fetch snapshots for up to ~100 symbols at once, using a simple cache."""
    if not symbols:
        return {}

    # Deduplicate and check cache
    unique_symbols = list(dict.fromkeys(symbols))  # preserve order, drop dupes
    cached: dict[str, dict] = {}
    to_fetch: list[str] = []
    for sym in unique_symbols:
        snap = _cache_get(sym)
        if snap is not None:
            cached[sym] = snap
        else:
            to_fetch.append(sym)

    results: dict[str, dict] = dict(cached)  # start with cached results

    if not to_fetch:
        return results

    # Batch into groups of 50 to stay within URL limits
    BATCH = 50
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i in range(0, len(to_fetch), BATCH):
            batch = to_fetch[i : i + BATCH]
            try:
                resp = await client.get(
                    f"{_ALPACA_BASE}/v2/options/snapshots",
                    params={"symbols": ",".join(batch), "feed": "indicative"},
                    headers=_alpaca_headers(),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    batch_snapshots = data.get("snapshots") or {}
                    for sym, snap in batch_snapshots.items():
                        results[sym] = snap
                        _cache_set(sym, snap)
                # Non‑200 responses are ignored (best‑effort)
            except Exception:
                # Silently ignore failures; callers will receive contracts without Greeks
                continue
    return results


@router.get("/chain/{symbol}")
async def get_options_chain(
    symbol: str,
    expiration: str | None = Query(
        None,
        description="Filter to a single expiration date YYYY-MM-DD",
        example="2024-01-21",
    ),
    strike_min: float | None = Query(
        None,
        description="Minimum strike price",
        example=100.0,
    ),
    strike_max: float | None = Query(
        None,
        description="Maximum strike price",
        example=200.0,
    ),
    option_type: Literal["call", "put", "all"] = Query("all", description="Option type filter"),
    current_user: User = Depends(get_current_user),
):
    """Fetch and enrich an options chain for a given underlying symbol."""
    # Return empty data with a helpful message when broker credentials are not configured
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        return {
            "contracts": [],
            "message": "Configure Alpaca API credentials to enable options chain data",
        }

    today = date.today().isoformat()

    params: dict[str, str | int] = {
        "underlying_symbols": symbol.upper(),
        "limit": 200,
    }
    if expiration:
        params["expiration_date_gte"] = expiration
        params["expiration_date_lte"] = expiration
    else:
        params["expiration_date_gte"] = today

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(
                f"{_ALPACA_BASE}/v2/options/contracts",
                params=params,
                headers=_alpaca_headers(),
            )
        except httpx.RequestError as exc:
            raise HTTPException(502, f"Alpaca connection error: {exc}") from exc

    if resp.status_code == 403:
        return {
            "contracts": [],
            "message": "Configure TradeStation API credentials to enable options data",
        }
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Alpaca API error: {resp.text[:200]}")

    data = resp.json()
    contracts: list[dict] = data.get("option_contracts") or []

    # Apply optional filters
    if option_type != "all":
        contracts = [c for c in contracts if c.get("type") == option_type]
    if strike_min is not None:
        contracts = [
            c
            for c in contracts
            if c.get("strike_price") is not None and float(c["strike_price"]) >= strike_min
        ]
    if strike_max is not None:
        contracts = [
            c
            for c in contracts
            if c.get("strike_price") is not None and float(c["strike_price"]) <= strike_max
        ]

    # Early exit if no contracts after filtering
    if not contracts:
        return []

    # Fetch snapshots (Greeks + quotes) for all filtered contracts
    symbols_list = [c["symbol"] for c in contracts if c.get("symbol")]
    snapshots = await _fetch_snapshots(symbols_list)

    enriched_dicts = [_enrich_contract(c, snapshots.get(c.get("symbol", ""))) for c in contracts]
    enriched: List[EnrichedOptionContract] = [EnrichedOptionContract(**d) for d in enriched_dicts]

    return enriched


@router.get("/snapshot/{symbol}")
async def get_options_snapshot(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    """Fetch latest Greeks snapshot for a single options contract symbol."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                f"{_ALPACA_BASE}/v2/options/snapshots",
                params={"symbols": symbol.upper(), "feed": "indicative"},
                headers=_alpaca_headers(),
            )
        except httpx.RequestError as exc:
            raise HTTPException(502, f"Alpaca connection error: {exc}") from exc

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Alpaca API error: {resp.text[:200]}")

    data = resp.json()
    snapshots = data.get("snapshots") or {}
    snapshot = snapshots.get(symbol.upper())
    if not snapshot:
        return {"symbol": symbol.upper(), "message": "No snapshot data available"}

    # Return the raw snapshot; callers can map to EnrichedOptionContract if desired
    return snapshot