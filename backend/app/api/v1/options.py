"""Options trading endpoints: chain, snapshots, expirations.

Proxies Alpaca's options API. Uses settings.alpaca_api_key and
settings.alpaca_secret_key directly for market data (no per-account
credentials needed).
"""
from __future__ import annotations

from datetime import date
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_current_user
from app.config import settings
from app.models.user import User

router = APIRouter(prefix="/options", tags=["options"])

_ALPACA_BASE = "https://paper-api.alpaca.markets"


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


async def _fetch_snapshots(symbols: list[str]) -> dict[str, dict]:
    """Fetch snapshots for up to ~100 symbols at once."""
    if not symbols:
        return {}
    # Batch into groups of 50 to stay within URL limits
    BATCH = 50
    results: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i in range(0, len(symbols), BATCH):
            batch = symbols[i : i + BATCH]
            try:
                resp = await client.get(
                    f"{_ALPACA_BASE}/v2/options/snapshots",
                    params={"symbols": ",".join(batch), "feed": "indicative"},
                    headers=_alpaca_headers(),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results.update(data.get("snapshots") or {})
            except Exception:
                pass  # snapshots are best-effort; return contract data without Greeks
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
        raise HTTPException(403, "Alpaca options data requires an approved options account level.")
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Alpaca API error: {resp.text[:200]}")

    data = resp.json()
    contracts: list[dict] = data.get("option_contracts") or []

    # Apply optional filters
    if option_type != "all":
        contracts = [c for c in contracts if c.get("type") == option_type]
    if strike_min is not None:
        contracts = [c for c in contracts if c.get("strike_price") is not None and float(c["strike_price"]) >= strike_min]
    if strike_max is not None:
        contracts = [c for c in contracts if c.get("strike_price") is not None and float(c["strike_price"]) <= strike_max]

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
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                f"{_ALPACA_BASE}/v2/options/snapshots",
                params={"symbols": symbol.upper(), "feed": "indicative"},
                headers=_alpaca_headers(),
            )
        except httpx.RequestError as exc:
            raise HTTPException(502, f"Alpaca connection error: {exc}") from exc

    if resp.status_code == 403:
        raise HTTPException(403, "Alpaca options data requires an approved options account level.")
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Alpaca API error: {resp.text[:200]}")

    data = resp.json()
    snapshots: dict = data.get("snapshots") or {}
    snap = snapshots.get(symbol.upper())
    if snap is None:
        raise HTTPException(404, f"No snapshot found for {symbol}")
    return snap


@router.get("/expirations/{underlying}")
async def get_options_expirations(
    underlying: str,
    current_user: User = Depends(get_current_user),
):
    """Return sorted list of distinct upcoming expiration dates for an underlying."""
    today = date.today().isoformat()
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(
                f"{_ALPACA_BASE}/v2/options/contracts",
                params={
                    "underlying_symbols": underlying.upper(),
                    "expiration_date_gte": today,
                    "limit": 200,
                },
                headers=_alpaca_headers(),
            )
        except httpx.RequestError as exc:
            raise HTTPException(502, f"Alpaca connection error: {exc}") from exc

    if resp.status_code == 403:
        raise HTTPException(403, "Alpaca options data requires an approved options account level.")
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Alpaca API error: {resp.text[:200]}")

    data = resp.json()
    contracts: list[dict] = data.get("option_contracts") or []
    dates = sorted({c["expiration_date"] for c in contracts if c.get("expiration_date")})
    return {"underlying": underlying.upper(), "expirations": dates}


# ── Legacy endpoints (kept for backward compatibility) ──────────────────────
from app.options.flow import scanner  # noqa: E402
from app.options.wheel import find_wheel_opportunities  # noqa: E402
from app.options.macro_calendar import get_upcoming_events, get_next_fomc  # noqa: E402


@router.get("/flow")
async def get_options_flow(
    unusual_only: bool = Query(False),
    current_user: User = Depends(get_current_user),
):
    flows = await scanner.scan()
    if unusual_only:
        flows = [f for f in flows if f.is_unusual]
    return [f.to_dict() for f in flows[:50]]


@router.get("/put-call-ratio")
async def get_put_call_ratio(current_user: User = Depends(get_current_user)):
    await scanner.scan()
    return scanner.put_call_ratio()


@router.get("/wheel")
async def get_wheel_opportunities(
    tickers: str = Query(None, description="Comma-separated tickers, e.g. AAPL,MSFT"),
    current_user: User = Depends(get_current_user),
):
    ticker_list = tickers.split(",") if tickers else None
    return [s.to_dict() for s in find_wheel_opportunities(ticker_list)]


@router.get("/macro-calendar")
async def get_macro_calendar(
    days_ahead: int = Query(90, le=365),
    current_user: User = Depends(get_current_user),
):
    return get_upcoming_events(days_ahead)


@router.get("/next-fomc")
async def next_fomc():
    """Public endpoint — no auth required for next FOMC date."""
    return get_next_fomc()
