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
    # Return empty data with a helpful message when broker credentials are not configured
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        return {
            "expirations": [],
            "message": "Configure TradeStation API credentials to enable options data",
        }

    today = date.today().isoformat()
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(
                f"{_ALPACA_BASE}/v2/options/contracts",
                params={
                    "underlying_symbols": underlying.upper(),
                    "limit": 200,
                    "expiration_date_gte": today,
                },
                headers=_alpaca_headers(),
            )
        except httpx.RequestError as exc:
            raise HTTPException(502, f"Alpaca connection error: {exc}") from exc

    if resp.status_code == 403:
        return {
            "expirations": [],
            "message": "Configure TradeStation API credentials to enable options data",
        }
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Alpaca API error: {resp.text[:200]}")

    data = resp.json()
    contracts: list[dict] = data.get("option_contracts") or []

    expirations = sorted(
        {c["expiration_date"] for c in contracts if c.get("expiration_date")}
    )
    return {"expirations": expirations}


# ═══════════════════════════════════════════════════════════════════════════
# Option Alpha-style automation: rules validation, flow, wheel, macro calendar
# ═══════════════════════════════════════════════════════════════════════════

_ALPACA_DATA = "https://data.alpaca.markets"
_DEFAULT_EQUITY = 100_000.0  # paper fallback when no account row exists
_MAX_RISK_PCT = 0.05         # Option Alpha guideline: ≤5% of equity per position

# Federal Reserve published meeting schedule (decision day = second day).
# 2027 dates are the Fed's tentative schedule.
_FOMC_DECISION_DATES = [
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]

# Per-underlying options snapshot cache (one API call covers a whole chain)
_UND_SNAP_TTL = 60.0
_und_snap_cache: Dict[str, Tuple[float, dict]] = {}

_FLOW_WATCHLIST = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT"]


def _parse_occ(occ: str) -> tuple[str, date, str, float] | None:
    """Parse an OCC option symbol, e.g. SPY260918P00560000."""
    try:
        root = occ[:-15]
        exp = date(2000 + int(occ[-15:-13]), int(occ[-13:-11]), int(occ[-11:-9]))
        cp = "call" if occ[-9] == "C" else "put"
        strike = int(occ[-8:]) / 1000.0
        if not root or strike <= 0:
            return None
        return root, exp, cp, strike
    except Exception:
        return None


async def _fetch_underlying_snapshots(sym: str) -> dict[str, dict]:
    """All option snapshots for one underlying in a single data-API call."""
    now = time.time()
    hit = _und_snap_cache.get(sym)
    if hit and now - hit[0] < _UND_SNAP_TTL:
        return hit[1]
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        return {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_ALPACA_DATA}/v1beta1/options/snapshots/{sym.upper()}",
                params={"feed": "indicative", "limit": 1000},
                headers=_alpaca_headers(),
            )
        if resp.status_code != 200:
            return {}
        snaps = resp.json().get("snapshots") or {}
        _und_snap_cache[sym] = (now, snaps)
        return snaps
    except Exception:
        return {}


def _iv_percentile(iv: float | None, universe: list[float]) -> float | None:
    """Rank an IV within the fetched chain's IVs (cross-sectional proxy for IV rank)."""
    if iv is None or not universe:
        return None
    below = sum(1 for x in universe if x <= iv)
    return round(100.0 * below / len(universe), 1)


# ── IV history → true IV rank ────────────────────────────────────────────────
# The cross-sectional proxy above ranks a contract against today's chain, which
# is NOT what traders mean by "IV rank" (current IV vs its own 52-week range).
# Every chain fetch upserts one {date: median-chain-IV} point per underlying
# into Redis; once ~a month of dailies accumulates, the wheel screener switches
# to genuine time-series rank. Falls back to the proxy while history is thin
# (and entirely no-ops when Redis is disabled).

_IV_HIST_MAX_DAYS = 252          # one trading year
_IV_HIST_MIN_POINTS = 20         # below this, history rank is meaningless
_IV_HIST_TTL = 60 * 60 * 24 * 400  # Redis key TTL: refreshed on every write


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _iv_rank_from_history(current_iv: float | None, history: dict[str, float]) -> float | None:
    """Percentile of current IV within its own trailing daily history.

    ``history`` maps ISO date → that day's median chain IV. Returns None until
    there are at least _IV_HIST_MIN_POINTS points — a thin series produces
    confident-looking nonsense.
    """
    if current_iv is None or len(history) < _IV_HIST_MIN_POINTS:
        return None
    values = sorted(history.values())
    below = sum(1 for v in values if v <= current_iv)
    return round(100.0 * below / len(values), 1)


async def _record_and_load_iv_history(ticker: str, snaps: dict[str, dict]) -> dict[str, float]:
    """Upsert today's median chain IV for a ticker and return the full history."""
    try:
        from app.redis_client import price_cache

        key = f"iv_hist:{ticker.upper()}"
        raw = await price_cache.get(key)
        import json as _json

        history: dict[str, float] = _json.loads(raw) if raw else {}
        today = date.today().isoformat()
        if today not in history:
            ivs = [s.get("impliedVolatility") for s in snaps.values() if s.get("impliedVolatility")]
            med = _median([float(v) for v in ivs])
            if med is not None:
                history[today] = round(med, 5)
                if len(history) > _IV_HIST_MAX_DAYS:
                    for stale in sorted(history)[: len(history) - _IV_HIST_MAX_DAYS]:
                        del history[stale]
                await price_cache.set(key, _json.dumps(history), ttl=_IV_HIST_TTL)
        return history
    except Exception:
        return {}


class RulesValidationRequest(BaseModel):
    account_id: Optional[str] = None
    symbol: str
    option_symbol: str
    expiration_date: str
    side: Literal["buy", "sell"]
    quantity: int
    credit_received: float = 0.0
    delta: float = 0.0
    strategy_type: Literal["csp", "covered_call", "iron_condor", "long_call", "long_put"]


@router.post("/rules/validate")
async def validate_trade_rules(
    body: RulesValidationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Option Alpha-style pre-trade rules check.

    Deterministic — no LLM involved. Checks DTE band, delta band, and position
    size against account equity, and computes the standard automation exits:
    50%-of-credit profit target, 2x-credit stop, and the 21-DTE management date.
    """
    warnings: list[str] = []
    errors: list[str] = []

    try:
        exp = date.fromisoformat(body.expiration_date[:10])
    except ValueError:
        raise HTTPException(422, "expiration_date must be YYYY-MM-DD")
    dte = (exp - date.today()).days

    parsed = _parse_occ(body.option_symbol)
    strike = parsed[3] if parsed else None

    is_short_premium = body.side == "sell" or body.strategy_type in ("csp", "covered_call", "iron_condor")

    # Premium: trust the client's credit, else best-effort snapshot lookup
    premium = body.credit_received if body.credit_received > 0 else None
    if premium is None:
        snap = (await _fetch_snapshots([body.option_symbol])).get(body.option_symbol)
        if snap:
            lq = snap.get("latestQuote") or {}
            bid, ask = lq.get("bp"), lq.get("ap")
            if bid is not None and ask is not None:
                premium = round((bid + ask) / 2, 4)

    # ── DTE rule ──────────────────────────────────────────────────────────
    if is_short_premium:
        dte_target = "30–45"
        if dte < 7:
            dte_status = "error"; errors.append(f"{dte} DTE is too close to expiration for short premium")
        elif dte < 21 or dte > 60:
            dte_status = "warn"; warnings.append(f"{dte} DTE is outside the 30–45 sweet spot")
        else:
            dte_status = "ok"
    else:
        dte_target = "45+"
        if dte < 14:
            dte_status = "error"; errors.append(f"{dte} DTE gives a long option little time to work")
        elif dte < 30:
            dte_status = "warn"; warnings.append(f"{dte} DTE is short for a long option — theta burn is steep")
        else:
            dte_status = "ok"

    # ── Delta rule ────────────────────────────────────────────────────────
    abs_delta = abs(body.delta)
    if is_short_premium:
        delta_target = "≤ 0.30"
        if abs_delta == 0:
            delta_status = "warn"; warnings.append("Delta unavailable — cannot verify probability of profit")
        elif abs_delta > 0.40:
            delta_status = "error"; errors.append(f"|Δ| {abs_delta:.2f} is too directional for short premium")
        elif abs_delta > 0.30:
            delta_status = "warn"; warnings.append(f"|Δ| {abs_delta:.2f} above the 0.30 short-premium guideline")
        else:
            delta_status = "ok"
    else:
        delta_target = "0.50–0.80"
        if abs_delta == 0:
            delta_status = "warn"; warnings.append("Delta unavailable — cannot verify moneyness")
        elif abs_delta < 0.35:
            delta_status = "warn"; warnings.append(f"|Δ| {abs_delta:.2f} is a low-probability lottery ticket")
        else:
            delta_status = "ok"

    # ── IV rank rule (needs 52-week IV history we don't store — flag it) ──
    iv_rank_detail = {"value": None, "target": "≥ 30 for premium selling", "status": "warn"}
    if is_short_premium:
        warnings.append("IV rank unavailable — verify elevated IV before selling premium")
    else:
        iv_rank_detail["target"] = "≤ 50 for debit trades"

    # ── Position size rule vs account equity ─────────────────────────────
    equity = _DEFAULT_EQUITY
    try:
        q = select(Account).where(Account.user_id == current_user.id, Account.is_active == True)  # noqa: E712
        if body.account_id:
            q = select(Account).where(Account.id == body.account_id, Account.user_id == current_user.id)
        acct = (await db.execute(q.limit(1))).scalar_one_or_none()
        if acct is not None and getattr(acct, "total_equity", None):
            equity = float(acct.total_equity)
    except Exception:
        pass

    if body.strategy_type == "csp" and strike:
        risk_per_contract = (strike - (premium or 0)) * 100
    elif premium:
        risk_per_contract = premium * 100
    else:
        risk_per_contract = None

    if risk_per_contract and risk_per_contract > 0:
        max_contracts = int((equity * _MAX_RISK_PCT) // risk_per_contract)
        if body.quantity > max_contracts:
            size_status = "error"
            errors.append(
                f"{body.quantity} contracts risks more than {_MAX_RISK_PCT:.0%} of equity "
                f"(max {max_contracts} at ~${risk_per_contract:,.0f}/contract)"
            )
        elif max_contracts and body.quantity > max_contracts * 0.8:
            size_status = "warn"; warnings.append("Position size close to the 5%-of-equity ceiling")
        else:
            size_status = "ok"
    else:
        max_contracts = None
        size_status = "warn"
        warnings.append("Cannot price risk per contract — size check skipped")

    # ── Automation exits (the Option Alpha standard playbook) ─────────────
    if is_short_premium and premium:
        profit_target_price = round(premium * 0.5, 2)          # buy back at 50% of credit
        stop_loss_price = round(premium * 2.0, 2)              # stop at 2x credit
        max_profit = round(premium * 100 * body.quantity, 2)
        max_loss_if_stopped = round(premium * 100 * body.quantity, 2)  # net loss at the 2x stop: pay back 2x, keep 1x
    elif premium:
        profit_target_price = round(premium * 2.0, 2)          # sell at 100% gain
        stop_loss_price = round(premium * 0.5, 2)              # stop at 50% of debit
        max_profit = round(premium * 100 * body.quantity, 2)   # profit at the 2x target
        max_loss_if_stopped = round(premium * 50 * body.quantity, 2)
    else:
        profit_target_price = stop_loss_price = max_loss_if_stopped = None
        max_profit = 0.0

    manage_days = 21 if is_short_premium else 7
    exit_before = exp - timedelta(days=manage_days)
    exit_before_date = exit_before.isoformat() if exit_before > date.today() else None

    return {
        "is_valid": not errors,
        "warnings": warnings,
        "errors": errors,
        "rules": {
            "dte": {"value": dte, "target": dte_target, "status": dte_status},
            "delta": {"value": round(abs_delta, 3) if abs_delta else None, "target": delta_target, "status": delta_status},
            "iv_rank": iv_rank_detail,
            "position_size": {"value": body.quantity, "target": f"≤ {_MAX_RISK_PCT:.0%} of equity", "status": size_status, "max": max_contracts},
        },
        "profit_target_price": profit_target_price,
        "stop_loss_price": stop_loss_price,
        "exit_before_date": exit_before_date,
        "max_profit": max_profit,
        "max_loss_if_stopped": max_loss_if_stopped,
        "dte": dte,
    }


@router.get("/flow")
async def get_options_flow(
    unusual_only: bool = Query(False),
    current_user: User = Depends(get_current_user),
):
    """Today's options flow across the liquid watchlist, from Alpaca snapshots.

    premium = last trade price × day volume × 100 (dollar premium traded today).
    is_unusual = large dollar premium or heavy volume at elevated IV.
    Empty list when Alpaca credentials are missing — never fabricated rows.
    """
    snap_sets = await asyncio.gather(*(_fetch_underlying_snapshots(s) for s in _FLOW_WATCHLIST))

    rows: list[dict] = []
    for underlying, snaps in zip(_FLOW_WATCHLIST, snap_sets):
        ivs = [s.get("impliedVolatility") for s in snaps.values() if s.get("impliedVolatility") is not None]
        per_underlying: list[dict] = []
        for occ, snap in snaps.items():
            parsed = _parse_occ(occ)
            if not parsed:
                continue
            _, exp, cp, strike = parsed
            daily = snap.get("dailyBar") or snap.get("minuteBar") or {}
            volume = float(daily.get("v") or 0)
            lt = snap.get("latestTrade") or {}
            last = lt.get("p")
            if not volume or last is None:
                continue
            iv = snap.get("impliedVolatility")
            iv_pct = _iv_percentile(iv, ivs)
            premium = round(last * volume * 100, 2)
            per_underlying.append({
                "ticker": underlying,
                "option_type": cp,
                "strike": strike,
                "expiry": exp.isoformat(),
                "volume": int(volume),
                "last": last,
                "iv": iv,
                "iv_percentile": iv_pct,
                "premium": premium,
                "is_unusual": premium > 250_000 or (volume > 1_000 and (iv_pct or 0) > 80),
            })
        per_underlying.sort(key=lambda r: r["premium"], reverse=True)
        rows.extend(per_underlying[:10])

    rows.sort(key=lambda r: r["premium"], reverse=True)
    if unusual_only:
        rows = [r for r in rows if r["is_unusual"]]
    return rows


@router.get("/put-call-ratio")
async def get_put_call_ratio(
    symbol: str = Query("SPY"),
    current_user: User = Depends(get_current_user),
):
    """Dashboard-shaped PCR: delegates to the real /market-data/pcr computation."""
    from app.api.v1.market_data import get_pcr

    data = await get_pcr(symbol=symbol, current_user=current_user)
    ratio = data.get("pcr")
    signal = data.get("signal") or "unavailable"
    return {
        "ratio": ratio,
        "puts": int(data.get("put_volume") or 0) or None,
        "calls": int(data.get("call_volume") or 0) or None,
        "sentiment": signal if signal in ("bullish", "bearish", "neutral") else "unavailable",
        "symbol": data.get("symbol"),
        "source": data.get("source"),
    }


@router.get("/wheel")
async def get_wheel_candidates(
    tickers: str = Query("AAPL,MSFT,NVDA,AMD,SPY"),
    current_user: User = Depends(get_current_user),
):
    """Cash-secured-put screener: ~0.16–0.35Δ puts, 21–49 DTE, best annualized yield.

    iv_rank is the contract's IV percentile within its own chain (cross-sectional
    proxy — true 52-week IV rank needs history we don't store).
    """
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()][:8]
    snap_sets = await asyncio.gather(*(_fetch_underlying_snapshots(s) for s in symbols))

    results: list[dict] = []
    today = date.today()
    for underlying, snaps in zip(symbols, snap_sets):
        iv_history = await _record_and_load_iv_history(underlying, snaps) if snaps else {}
        put_ivs = []
        candidates: list[dict] = []
        for occ, snap in snaps.items():
            parsed = _parse_occ(occ)
            if not parsed:
                continue
            _, exp, cp, strike = parsed
            if cp != "put":
                continue
            iv = snap.get("impliedVolatility")
            if iv is not None:
                put_ivs.append(iv)
            dte = (exp - today).days
            if not (21 <= dte <= 49):
                continue
            greeks = snap.get("greeks") or {}
            delta = greeks.get("delta")
            if delta is None or not (-0.35 <= delta <= -0.16):
                continue
            lq = snap.get("latestQuote") or {}
            bid, ask = lq.get("bp"), lq.get("ap")
            if not bid or ask is None:
                continue
            mid = (bid + ask) / 2
            if mid <= 0 or strike <= 0:
                continue
            candidates.append({
                "ticker": underlying,
                "strike": strike,
                "expiry": exp.isoformat(),
                "dte": dte,
                "premium": round(mid, 2),
                "delta": round(delta, 3),
                "iv": iv,
                "annualized_yield": round((mid / strike) * (365 / dte) * 100, 2),
            })
        for c in candidates:
            iv = c.pop("iv")
            hist_rank = _iv_rank_from_history(iv, iv_history)
            if hist_rank is not None:
                c["iv_rank"] = hist_rank
                c["iv_rank_source"] = "history"  # true trailing-252d rank
            else:
                c["iv_rank"] = _iv_percentile(iv, put_ivs) or 0.0
                c["iv_rank_source"] = "chain_proxy"  # cross-sectional until history accrues
        candidates.sort(key=lambda c: c["annualized_yield"], reverse=True)
        if candidates:
            results.append(candidates[0])

    results.sort(key=lambda c: c["annualized_yield"], reverse=True)
    return results


def _shift_to_weekday(d: date) -> date:
    """Roll a weekend date forward to Monday."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


@router.get("/macro-calendar")
async def get_macro_calendar(
    days_ahead: int = Query(60, ge=1, le=365),
    current_user: User = Depends(get_current_user),
):
    """Upcoming macro events: FOMC (published Fed schedule) plus rule-based
    estimates for NFP (first Friday), CPI (~13th), PPI (~14th), GDP (~28th of
    quarter-end months). Estimated dates are labeled as such."""
    today = date.today()
    horizon = today + timedelta(days=days_ahead)
    events: list[dict] = []

    def _add(d: date, category: str, title: str, importance: str) -> None:
        if today <= d <= horizon:
            events.append({
                "date": d.isoformat(),
                "days_away": (d - today).days,
                "category": category,
                "title": title,
                "importance": importance,
            })

    for iso in _FOMC_DECISION_DATES:
        _add(date.fromisoformat(iso), "fomc", "FOMC Rate Decision", "high")

    cursor = date(today.year, today.month, 1)
    while cursor <= horizon:
        first_friday = cursor + timedelta(days=(4 - cursor.weekday()) % 7)
        _add(first_friday, "nfp", "Nonfarm Payrolls", "high")
        _add(_shift_to_weekday(cursor.replace(day=13)), "cpi", "CPI Release (est.)", "high")
        _add(_shift_to_weekday(cursor.replace(day=14)), "ppi", "PPI Release (est.)", "medium")
        if cursor.month in (1, 4, 7, 10):
            _add(_shift_to_weekday(cursor.replace(day=28)), "gdp", "GDP Advance Estimate (est.)", "medium")
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)

    events.sort(key=lambda e: e["date"])
    return events


@router.get("/next-fomc")
async def get_next_fomc(current_user: User = Depends(get_current_user)):
    """Next FOMC decision date from the Fed's published meeting schedule."""
    today = date.today()
    for iso in _FOMC_DECISION_DATES:
        d = date.fromisoformat(iso)
        if d >= today:
            return {"date": iso, "days_away": (d - today).days}
    return {"date": None, "days_away": None}