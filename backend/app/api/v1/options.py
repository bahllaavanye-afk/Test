"""Options trading endpoints: chain, snapshots, expirations, rules validation.

Proxies Alpaca's options API. Uses settings.alpaca_api_key and
settings.alpaca_secret_key directly for market data (no per-account
credentials needed).
"""
from __future__ import annotations

import asyncio
import math
import time
import re
from datetime import date, datetime, timezone
from typing import Literal, Optional, Dict, Tuple

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
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
        "underlying_symbol": contract.get("underlying_symbol") or contract.get("root_symbol"),
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


def _validate_symbol(symbol: str) -> None:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non‑empty string")
    if not re.fullmatch(r"[A-Za-z0-9\.\-]{1,20}", symbol):
        raise ValueError(f"symbol '{symbol}' contains invalid characters")


def _validate_expiration(expiration: str | None) -> None:
    if expiration is None:
        return
    try:
        datetime.strptime(expiration, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("expiration must be in YYYY‑MM‑DD format") from exc


def _validate_strike_range(strike_min: float | None, strike_max: float | None) -> None:
    if strike_min is not None and strike_min < 0:
        raise ValueError("strike_min must be non‑negative")
    if strike_max is not None and strike_max < 0:
        raise ValueError("strike_max must be non‑negative")
    if strike_min is not None and strike_max is not None and strike_min > strike_max:
        raise ValueError("strike_min cannot be greater than strike_max")


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
    expiration: str | None = Query(None, description="Filter to a single expiration date YYYY-MM-DD"),
    strike_min: float | None = Query(None, description="Minimum strike price"),
    strike_max: float | None = Query(None, description="Maximum strike price"),
    option_type: Literal["call", "put", "all"] = Query("all"),
    current_user: User = Depends(get_current_user),
):
    """Fetch and enrich an options chain for a given underlying symbol."""
    # Input validation
    _validate_symbol(symbol)
    _validate_expiration(expiration)
    _validate_strike_range(strike_min, strike_max)

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

    enriched = [_enrich_contract(c, snapshots.get(c.get("symbol", ""))) for c in contracts]
    return enriched


@router.get("/snapshot/{symbol}")
async def get_options_snapshot(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    """Fetch latest Greeks snapshot for a single options contract symbol."""
    # Input validation
    _validate_symbol(symbol)

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise HTTPException(
            400, "Alpaca API credentials are not configured; cannot fetch snapshot."
        )

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
    return snapshots.get(symbol.upper(), {})