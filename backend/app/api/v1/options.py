"""Options trading endpoints: chain, snapshots, expirations, rules validation.

Proxies Alpaca's options API. Uses settings.alpaca_api_key and
settings.alpaca_secret_key directly for market data (no per-account
credentials needed).
"""
from __future__ import annotations

import time
from datetime import date
from typing import Literal, Dict, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from app.api.deps import get_current_user
from app.config import settings
from app.models.user import User

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
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise HTTPException(
            403,
            "Configure Alpaca API credentials to enable options snapshot data",
        )

    symbol = symbol.upper()

    # Try cache first
    cached = _cache_get(symbol)
    if cached is not None:
        return cached

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                f"{_ALPACA_BASE}/v2/options/snapshots",
                params={"symbols": symbol, "feed": "indicative"},
                headers=_alpaca_headers(),
            )
        except httpx.RequestError as exc:
            raise HTTPException(502, f"Alpaca connection error: {exc}") from exc

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Alpaca API error: {resp.text[:200]}")

    data = resp.json()
    snapshots = data.get("snapshots") or {}
    snapshot = snapshots.get(symbol)

    if snapshot:
        _cache_set(symbol, snapshot)
        return snapshot

    # If Alpaca returns no data for the symbol, return an empty dict
    return {}