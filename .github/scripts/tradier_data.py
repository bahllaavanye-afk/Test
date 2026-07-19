#!/usr/bin/env python3
"""Tradier options data — real chains WITH ORATS-computed greeks/IV.

Verified live 2026-07-19 against the sandbox: SPY returned 33 expirations and
334 contracts, every one carrying greeks (delta/mid_iv/theta/gamma/vega). This
is the free real-greeks unlock the options desk needed — it lets the desk pick
strikes by ACTUAL delta (not a moneyness proxy) and read real IV instead of the
HV estimate.

Auth: TRADIER_SANDBOX_TOKEN (sandbox, paper) preferred; TRADIER_TOKEN for prod.
Endpoint follows the token: sandbox token -> sandbox.tradier.com, else api.tradier.com.
Everything is fail-soft — a feed hiccup returns None/[] and the caller keeps its
existing fallback (moneyness strikes / HV proxy). Never raises.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date, datetime

_SANDBOX_TOKEN = os.environ.get("TRADIER_SANDBOX_TOKEN", "").strip()
_PROD_TOKEN = os.environ.get("TRADIER_TOKEN", "").strip()
TOKEN = _SANDBOX_TOKEN or _PROD_TOKEN
BASE = "https://sandbox.tradier.com/v1" if _SANDBOX_TOKEN or not _PROD_TOKEN else "https://api.tradier.com/v1"

_UA = "QuantEdge-TradierAdapter/1.0"


def available() -> bool:
    return bool(TOKEN)


def _get(path: str, timeout: int = 20):
    """GET a Tradier endpoint as JSON, or None on any failure (never raises)."""
    if not TOKEN:
        return None
    try:
        req = urllib.request.Request(
            f"{BASE}{path}",
            headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json", "User-Agent": _UA},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001 — feed down -> caller keeps its fallback
        print(f"  ⚠ tradier {path[:48]} failed: {str(exc)[:80]}", flush=True)
        return None


def quote(symbol: str) -> dict | None:
    """{last, bid, ask} for an underlying, or None."""
    d = _get(f"/markets/quotes?symbols={urllib.parse.quote(symbol)}")
    if not d:
        return None
    q = (d.get("quotes") or {}).get("quote")
    if isinstance(q, list):
        q = q[0] if q else None
    if not q:
        return None
    return {"last": q.get("last"), "bid": q.get("bid"), "ask": q.get("ask")}


def expirations(symbol: str) -> list[str]:
    """Sorted list of expiration dates (YYYY-MM-DD), or []."""
    d = _get(f"/markets/options/expirations?symbol={urllib.parse.quote(symbol)}")
    if not d:
        return []
    dates = (d.get("expirations") or {}).get("date") or []
    if isinstance(dates, str):
        dates = [dates]
    return sorted(dates)


def nearest_expiration(symbol: str, target_dte: int) -> str | None:
    """The listed expiration whose DTE is closest to target_dte, or None."""
    exps = expirations(symbol)
    if not exps:
        return None
    today = date.today()
    def dte(e: str) -> int:
        return (datetime.strptime(e, "%Y-%m-%d").date() - today).days
    return min(exps, key=lambda e: abs(dte(e) - target_dte))


def chain(symbol: str, expiration: str, greeks: bool = True) -> list[dict]:
    """Full option chain for one expiration, each contract carrying greeks."""
    g = "true" if greeks else "false"
    d = _get(f"/markets/options/chains?symbol={urllib.parse.quote(symbol)}"
             f"&expiration={expiration}&greeks={g}")
    if not d:
        return []
    opts = (d.get("options") or {}).get("option") or []
    if isinstance(opts, dict):
        opts = [opts]
    return opts


def pick_by_delta(symbol: str, target_dte: int, target_delta: float,
                  right: str) -> dict | None:
    """Pick the contract closest to |delta| == target_delta on the nearest
    expiration. `right` is 'put' or 'call'. Returns the raw Tradier option dict
    (has 'symbol', 'strike', 'bid', 'ask', 'greeks') or None if unavailable.

    This is the real-delta replacement for the desk's moneyness strike guess.
    """
    exp = nearest_expiration(symbol, target_dte)
    if not exp:
        return None
    opts = [o for o in chain(symbol, exp, greeks=True)
            if (o.get("option_type") == right)]
    scored = []
    for o in opts:
        gr = o.get("greeks") or {}
        delta = gr.get("delta")
        if delta is None:
            continue
        scored.append((abs(abs(float(delta)) - abs(target_delta)), o))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    return scored[0][1]


def atm_iv(symbol: str, target_dte: int = 30) -> float | None:
    """At-the-money implied vol (mid_iv) for the nearest expiration, or None.

    Building block for a real IV-rank: persist this daily and rank the current
    value against the trailing window (replaces the HV proxy over time).
    """
    q = quote(symbol)
    exp = nearest_expiration(symbol, target_dte)
    if not q or not q.get("last") or not exp:
        return None
    spot = float(q["last"])
    opts = chain(symbol, exp, greeks=True)
    calls = [o for o in opts if o.get("option_type") == "call" and o.get("strike") is not None]
    if not calls:
        return None
    atm = min(calls, key=lambda o: abs(float(o["strike"]) - spot))
    iv = (atm.get("greeks") or {}).get("mid_iv")
    return float(iv) if iv is not None else None
