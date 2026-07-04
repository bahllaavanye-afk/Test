"""Options trading endpoints: chain, snapshots, expirations, rules validation.

Proxies Alpaca's options API. Uses settings.alpaca_api_key and
settings.alpaca_secret_key directly for market data (no per-account
credentials needed).
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import date, datetime, timezone
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


class OptionsContract(BaseModel):
    """A single options contract enriched with market data."""

    symbol: str = Field(..., description="Option contract symbol (e.g., AAPL240121C00150000)", example="AAPL240121C00150000")
    underlying_symbol: str = Field(..., description="Underlying equity symbol", example="AAPL")
    expiration_date: date = Field(..., description="Expiration date of the contract", example="2024-01-21")
    strike_price: float = Field(..., gt=0, description="Strike price of the option", example=150.0)
    option_type: Literal["call", "put"] = Field(..., description='Option type: "call" or "put"', example="call")
    bid: Optional[float] = Field(None, ge=0, description="Current best bid price", example=2.15)
    ask: Optional[float] = Field(None, ge=0, description="Current best ask price", example=2.45)
    mid: Optional[float] = Field(None, ge=0, description="Mid price between bid and ask", example=2.30)
    last: Optional[float] = Field(None, ge=0, description="Last traded price", example=2.40)
    volume: Optional[int] = Field(None, ge=0, description="Trading volume for the most recent session", example=1200)
    open_interest: Optional[int] = Field(None, ge=0, description="Open interest count", example=3500)
    implied_volatility: Optional[float] = Field(
        None,
        gt=0,
        description="Implied volatility as a decimal (e.g., 0.25 for 25%)",
        example=0.22,
    )
    delta: Optional[float] = Field(None, description="Delta Greek", example=0.55)
    gamma: Optional[float] = Field(None, description="Gamma Greek", example=0.12)
    theta: Optional[float] = Field(None, description="Theta Greek", example=-0.03)
    vega: Optional[float] = Field(None, description="Vega Greek", example=0.15)
    rho: Optional[float] = Field(None, description="Rho Greek", example=0.01)

    @validator("option_type")
    def check_option_type(cls, v: str) -> str:
        if v not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        return v

    @validator("expiration_date", pre=True)
    def parse_expiration(cls, v):
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v

    @validator("mid")
    def check_mid_between_bid_ask(cls, v, values):
        bid = values.get("bid")
        ask = values.get("ask")
        if v is not None and bid is not None and ask is not None:
            if not (bid <= v <= ask):
                raise ValueError("mid must be between bid and ask")
        return v


class OptionsSnapshot(BaseModel):
    """Snapshot data for a single option contract."""

    symbol: str = Field(..., description="Option contract symbol", example="AAPL240121C00150000")
    greeks: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Greek values keyed by name",
        example={"delta": 0.55, "gamma": 0.12, "theta": -0.03, "vega": 0.15, "rho": 0.01},
    )
    implied_volatility: Optional[float] = Field(
        None,
        gt=0,
        description="Implied volatility as a decimal",
        example=0.22,
    )
    latest_quote: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Latest quote data with bid/ask prices",
        example={"bp": 2.15, "ap": 2.45},
    )
    latest_trade: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Latest trade information",
        example={"p": 2.40, "s": 1200},
    )

    @validator("implied_volatility")
    def iv_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("implied_volatility must be positive")
        return v


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
    # FastAPI will automatically serialize the list of dicts; the schema is provided for documentation purposes.
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
        raise HTTPException(404, f"No snapshot data found for symbol {symbol}")

    # Return a validated Pydantic model for consistency with documentation
    return OptionsSnapshot(**{
        "symbol": symbol.upper(),
        "greeks": snapshot.get("greeks") or {},
        "implied_volatility": snapshot.get("impliedVolatility"),
        "latest_quote": snapshot.get("latestQuote") or {},
        "latest_trade": snapshot.get("latestTrade") or {},
    })