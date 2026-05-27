"""Options trading endpoints: chain, snapshots, expirations, rules validation.

Proxies Alpaca's options API. Uses settings.alpaca_api_key and
settings.alpaca_secret_key directly for market data (no per-account
credentials needed).
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timezone
from typing import Literal, Optional

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


# ── Options Rules Validation ───────────────────────────────────────────────


class OptionsRulesRequest(BaseModel):
    account_id: Optional[str] = None
    symbol: str
    option_symbol: str
    expiration_date: str  # YYYY-MM-DD
    side: Literal["buy", "sell"]
    quantity: int = 1
    credit_received: float = 0.0  # per share (× 100 for contract value)
    delta: float = 0.0
    strategy_type: Literal["csp", "covered_call", "iron_condor", "long_call", "long_put"] = "csp"


async def _fetch_account_equity(account_id: str, current_user: User, db: AsyncSession) -> float:
    """Fetch account equity from Alpaca for a given account."""
    result = await db.execute(
        select(Account).where(Account.id == account_id, Account.user_id == current_user.id)
    )
    acct = result.scalar_one_or_none()
    if not acct or acct.broker != "alpaca" or not acct.encrypted_key:
        return 0.0
    try:
        from app.brokers.alpaca_orders import get_alpaca_account
        data = await get_alpaca_account(acct)
        return float(data.get("equity", 0))
    except Exception:
        return 0.0


async def _fetch_iv_rank(symbol: str) -> Optional[float]:
    """
    Compute a proxy IV rank from Alpaca historical bars using 52-week realized vol.
    Returns a value 0–100 representing where current IV sits vs 52-week range.
    Falls back to None on any error.
    """
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    start_dt = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                - __import__("datetime").timedelta(days=380)).strftime("%Y-%m-%d")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://data.alpaca.markets/v2/stocks/bars",
                headers=headers,
                params={
                    "symbols": symbol.upper(),
                    "timeframe": "1Day",
                    "start": start_dt,
                    "feed": "iex",
                    "adjustment": "raw",
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            bars = data.get("bars", {}).get(symbol.upper(), [])
            if len(bars) < 20:
                return None

            closes = [float(b["c"]) for b in bars]

            # Rolling 30-day realized vol windows across 52 weeks
            window = 21  # ~1 month
            rv_list = []
            for i in range(window, len(closes)):
                window_closes = closes[i - window: i]
                returns = [math.log(window_closes[j] / window_closes[j - 1])
                           for j in range(1, len(window_closes))]
                mean_r = sum(returns) / len(returns)
                variance = sum((r - mean_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
                rv = math.sqrt(variance * 252) * 100  # annualised %
                rv_list.append(rv)

            if not rv_list:
                return None

            current_rv = rv_list[-1]
            min_rv = min(rv_list)
            max_rv = max(rv_list)
            if max_rv == min_rv:
                return 50.0
            iv_rank = (current_rv - min_rv) / (max_rv - min_rv) * 100.0
            return round(iv_rank, 1)
    except Exception:
        return None


@router.post("/rules/validate")
async def validate_options_rules(
    body: OptionsRulesRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Validate an options trade against professional Options Alpha risk rules.
    Returns pass/warn/fail status per rule plus calculated exit levels.
    """
    # Fetch account equity if account_id provided
    account_equity = 0.0
    if body.account_id:
        account_equity = await _fetch_account_equity(body.account_id, current_user, db)

    # ── DTE check ─────────────────────────────────────────────────────────
    today = date.today()
    try:
        exp_date = date.fromisoformat(body.expiration_date)
        dte_value = max(0, (exp_date - today).days)
    except ValueError:
        dte_value = 0

    if dte_value < 21:
        dte_status = "error"
    elif dte_value < 30:
        dte_status = "warn"
    elif dte_value <= 45:
        dte_status = "ok"
    elif dte_value <= 60:
        dte_status = "warn"
    else:
        dte_status = "warn"

    # ── Delta check ────────────────────────────────────────────────────────
    abs_delta = abs(body.delta)
    is_selling = body.side == "sell"
    if is_selling and body.strategy_type in ("csp", "covered_call"):
        if abs_delta > 0.35:
            delta_status = "error"
        elif abs_delta < 0.10:
            delta_status = "warn"
        elif 0.20 <= abs_delta <= 0.30:
            delta_status = "ok"
        else:
            delta_status = "warn"
    else:
        # For long strategies, delta is a feature not a constraint
        delta_status = "ok"

    # ── IV Rank check ──────────────────────────────────────────────────────
    iv_rank = await _fetch_iv_rank(body.symbol)
    if iv_rank is None:
        iv_rank_status = "warn"
    elif iv_rank < 30:
        iv_rank_status = "error" if is_selling else "ok"
    elif iv_rank < 50:
        iv_rank_status = "warn" if is_selling else "ok"
    else:
        iv_rank_status = "ok"

    # ── Position size check ────────────────────────────────────────────────
    contract_value = body.credit_received * 100.0
    if account_equity > 0 and contract_value > 0:
        max_quantity = int(account_equity * 0.05 / contract_value)
    else:
        max_quantity = 0  # unknown

    if max_quantity > 0:
        position_size_status = "ok" if body.quantity <= max_quantity else "error"
    else:
        position_size_status = "warn"

    # ── Build rules summary ────────────────────────────────────────────────
    rules = {
        "dte": {
            "value": dte_value,
            "target": "30-45",
            "status": dte_status,
        },
        "delta": {
            "value": round(abs_delta, 3),
            "target": "0.20-0.30" if is_selling else "any",
            "status": delta_status,
        },
        "iv_rank": {
            "value": iv_rank,
            "target": ">50",
            "status": iv_rank_status,
        },
        "position_size": {
            "value": body.quantity,
            "max": max_quantity if max_quantity > 0 else None,
            "status": position_size_status,
        },
    }

    # ── Warnings and errors ────────────────────────────────────────────────
    warnings = []
    errors = []

    if dte_status == "warn":
        if dte_value < 30:
            warnings.append(f"DTE is {dte_value} — below optimal 30-45 range")
        else:
            warnings.append(f"DTE is {dte_value} — above optimal 30-45 range")
    elif dte_status == "error":
        errors.append(f"DTE is {dte_value} — too close to expiration (< 21 days), high gamma risk")

    if delta_status == "warn":
        warnings.append(f"|Delta| is {abs_delta:.2f} — outside optimal 0.20-0.30 range")
    elif delta_status == "error":
        errors.append(f"|Delta| is {abs_delta:.2f} — too aggressive (> 0.35), high assignment risk")

    if iv_rank_status == "warn" and is_selling:
        warnings.append(f"IV Rank is {iv_rank} — below optimal >50 for premium selling")
    elif iv_rank_status == "error" and is_selling:
        errors.append(f"IV Rank is {iv_rank} — too low (< 30), premium is cheap, poor selling environment")
    elif iv_rank is None:
        warnings.append("IV Rank unavailable — could not fetch historical volatility data")

    if position_size_status == "warn":
        warnings.append("Account equity unavailable — position size not validated")
    elif position_size_status == "error":
        errors.append(
            f"Quantity {body.quantity} exceeds max {max_quantity} contracts "
            f"(5% of ${account_equity:,.0f} equity @ ${contract_value:.0f}/contract)"
        )

    # ── Exit levels ────────────────────────────────────────────────────────
    profit_target_price = round(body.credit_received * 0.50, 2) if body.credit_received > 0 else None
    stop_loss_price = round(body.credit_received * 2.00, 2) if body.credit_received > 0 else None
    max_profit = round(body.credit_received * 100.0 * body.quantity, 2)
    max_loss_if_stopped = round(-stop_loss_price * 100.0 * body.quantity, 2) if stop_loss_price else None

    # 21-DTE exit date
    exit_before_date = None
    if exp_date:
        exit_dt = exp_date - __import__("datetime").timedelta(days=21)
        exit_before_date = exit_dt.isoformat()

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "warnings": warnings,
        "errors": errors,
        "rules": rules,
        "profit_target_price": profit_target_price,
        "stop_loss_price": stop_loss_price,
        "exit_before_date": exit_before_date,
        "max_profit": max_profit,
        "max_loss_if_stopped": max_loss_if_stopped,
        "account_equity": round(account_equity, 2),
        "dte": dte_value,
    }


# ── Options Alpha Scanner ──────────────────────────────────────────────────

_ALPACA_DATA_BASE = "https://data.alpaca.markets"


def _data_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        "accept": "application/json",
    }


async def _fetch_symbol_bars(client: httpx.AsyncClient, symbol: str) -> list[dict]:
    """Fetch up to 252 daily bars for a symbol from Alpaca data API."""
    try:
        resp = await client.get(
            f"{_ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars",
            headers=_data_headers(),
            params={
                "timeframe": "1Day",
                "limit": 252,
                "feed": "iex",
                "adjustment": "raw",
            },
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("bars", [])
    except Exception:
        return []


def _compute_scanner_metrics(bars: list[dict]) -> dict | None:
    """Compute IV rank proxy, RSI-14, HV-20 from daily bars."""
    if len(bars) < 30:
        return None

    closes = np.array([float(b["c"]) for b in bars], dtype=np.float64)
    current_price = float(closes[-1])

    # Log returns
    log_ret = np.diff(np.log(closes))

    # HV-20 (annualised)
    hv_20 = float(log_ret[-20:].std() * np.sqrt(252)) if len(log_ret) >= 20 else None
    if hv_20 is None:
        return None

    # IV rank proxy using full history
    n = len(log_ret)
    if n >= 252:
        hv_series = np.array([
            log_ret[max(0, i - 20): i].std() * np.sqrt(252)
            for i in range(20, n + 1)
        ])
        hv_min = float(hv_series.min())
        hv_max = float(hv_series.max())
        iv_rank = float((hv_20 - hv_min) / max(hv_max - hv_min, 0.001) * 100)

        # IV percentile: fraction of days where HV was below current HV20
        iv_percentile = float((hv_series < hv_20).mean() * 100)
    else:
        iv_rank = 50.0
        iv_percentile = 50.0

    # RSI-14
    diff = np.diff(closes)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    if len(gains) >= 14:
        avg_gain = float(gains[-14:].mean())
        avg_loss = float(losses[-14:].mean())
        rsi_14 = float(100 - 100 / (1 + avg_gain / max(avg_loss, 1e-10)))
    else:
        rsi_14 = 50.0

    return {
        "current_price": round(current_price, 2),
        "iv_rank": round(iv_rank, 1),
        "iv_percentile": round(iv_percentile, 1),
        "hv_20": round(hv_20, 4),
        "rsi_14": round(rsi_14, 1),
    }


def _determine_best_strategy(iv_rank: float, rsi_14: float) -> tuple[str, str, str, str | None]:
    """Return (best_strategy, signal_strength, trade_signal, avoid_reason)."""
    avoid_reason: str | None = None

    # RSI extremes warn about trend direction
    if rsi_14 > 75 or rsi_14 < 25:
        # Trending market — iron condor has higher failure rate
        if iv_rank > 70:
            return "earnings_iv_crush", "moderate", "sell_premium", "strong_trend"
        elif iv_rank > 50:
            return "covered_call" if rsi_14 > 75 else "cash_secured_put", "moderate", "sell_premium", "strong_trend"
        elif rsi_14 < 25:
            return "long_call", "moderate", "buy_premium", None
        else:
            return "long_put", "moderate", "buy_premium", None

    if iv_rank > 70:
        best = "iron_condor"
        strength = "strong" if iv_rank > 80 else "moderate"
        trade_signal = "sell_premium"
    elif iv_rank > 50:
        # Good premium but not extreme — prefer directional theta plays
        best = "covered_call" if rsi_14 >= 50 else "cash_secured_put"
        strength = "moderate"
        trade_signal = "sell_premium"
    elif iv_rank > 30:
        best = "wheel"
        strength = "moderate" if iv_rank > 40 else "weak"
        trade_signal = "sell_premium"
    else:
        # IV cheap — buy premium instead of selling
        best = "long_call" if rsi_14 >= 50 else "long_put"
        strength = "weak"
        trade_signal = "buy_premium"

    return best, strength, trade_signal, avoid_reason


@router.get("/scanner")
async def options_scanner(
    symbols: str = Query(
        "AAPL,MSFT,NVDA,SPY,QQQ,TSLA,AMZN,META",
        description="Comma-separated list of underlying symbols to scan",
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Options Alpha Scanner — ranks symbols by IV rank and recommends the best
    options strategy for each.

    For each symbol:
    1. Fetches 252 days of daily bars from Alpaca
    2. Calculates IV Rank (HV20 percentile proxy)
    3. Checks RSI-14 for trend direction
    4. Recommends best strategy based on IV regime

    Returns list sorted by iv_rank descending.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(400, "No symbols provided")

    # Fetch all symbols concurrently
    async with httpx.AsyncClient(timeout=20.0) as client:
        bar_results = await asyncio.gather(
            *[_fetch_symbol_bars(client, sym) for sym in symbol_list],
            return_exceptions=True,
        )

    output = []
    for sym, bars in zip(symbol_list, bar_results):
        if isinstance(bars, Exception) or not bars:
            continue

        metrics = _compute_scanner_metrics(bars)
        if metrics is None:
            continue

        iv_rank = metrics["iv_rank"]
        rsi_14 = metrics["rsi_14"]

        # Determine IV regime label
        if iv_rank >= 70:
            regime = "high_iv"
        elif iv_rank >= 50:
            regime = "elevated_iv"
        elif iv_rank >= 30:
            regime = "normal_iv"
        else:
            regime = "low_iv"

        best_strategy, signal_strength, trade_signal, avoid_reason = _determine_best_strategy(
            iv_rank, rsi_14
        )

        output.append({
            "symbol": sym,
            "current_price": metrics["current_price"],
            "iv_rank": iv_rank,
            "iv_percentile": metrics["iv_percentile"],
            "hv_20": metrics["hv_20"],
            "rsi_14": rsi_14,
            "regime": regime,
            "best_strategy": best_strategy,
            "signal_strength": signal_strength,
            "trade_signal": trade_signal,
            "avoid_reason": avoid_reason,
        })

    # Sort by IV rank descending (highest premium opportunity first)
    output.sort(key=lambda x: x["iv_rank"], reverse=True)
    return output


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
