#!/usr/bin/env python3
"""Polymarket real market data for the desk — public Gamma + CLOB APIs.

The Polymarket desk ran its 8 poly_* strategies against an SPY proxy because
"no Polymarket data was wired". Both data endpoints are PUBLIC (verified
2026-07-18: top market returned 69 hourly points):

  gamma-api.polymarket.com/markets      — active markets ranked by 24h volume
  clob.polymarket.com/prices-history    — hourly price series per outcome token

This module feeds the desk real prediction-market prices. ORDER PLACEMENT
still needs py-clob-client + POLYMARKET_PRIVATE_KEY signing (keys are in the
secret relay; wiring queued) — until then poly signals are logged, not traded,
which is exactly what the desk already does for signal-only venues.
"""
from __future__ import annotations

import json
import urllib.request

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
_UA = "Mozilla/5.0 (X11; Linux x86_64) QuantEdge-PolyDesk/1.0"


def _get(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_top_markets(limit: int = 6) -> list[dict]:
    """Highest-24h-volume active markets: [{question, token_id, volume24h}]."""
    try:
        markets = _get(f"{GAMMA}/markets?active=true&closed=false"
                       f"&limit={limit}&order=volume24hr&ascending=false")
    except Exception as exc:  # noqa: BLE001 — feed down -> caller keeps proxy
        print(f"  ⚠ polymarket gamma fetch failed: {str(exc)[:80]}", flush=True)
        return []
    out = []
    for m in markets if isinstance(markets, list) else []:
        try:
            token_id = json.loads(m["clobTokenIds"])[0]   # YES outcome token
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "question": str(m.get("question", "?")),
            "token_id": str(token_id),
            "volume24h": float(m.get("volume24hr") or 0),
        })
    return out


def history_to_bars(points: list[dict]):
    """Hourly {t, p} points → OHLCV frame the strategies understand (pure).

    Prices are probabilities in [0,1]; scaled ×100 so percent-based indicator
    math (ATR thresholds etc.) sees sane magnitudes. OHLC is synthesized from
    consecutive prices; volume is a constant placeholder (the strategies used
    on this desk don't gate on volume)."""
    import pandas as pd
    if len(points) < 30:
        return None
    df = pd.DataFrame(points)
    if not {"t", "p"}.issubset(df.columns):
        return None
    df["close"] = df["p"].astype(float) * 100.0
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume"] = 1_000_000.0
    df.index = pd.to_datetime(df["t"].astype(int), unit="s", utc=True)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_price_bars(token_id: str):
    """One market's hourly bars over the last week, or None."""
    try:
        hist = _get(f"{CLOB}/prices-history?market={token_id}&interval=1w&fidelity=60")
        return history_to_bars(hist.get("history", []))
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ polymarket history failed for {token_id[:16]}…: {str(exc)[:80]}", flush=True)
        return None


def desk_feed(limit: int = 6) -> dict:
    """{symbol: bars} for the desk — symbol is 'PM:<question snippet>'."""
    feed: dict = {}
    for m in fetch_top_markets(limit):
        bars = fetch_price_bars(m["token_id"])
        if bars is None:
            continue
        sym = "PM:" + m["question"][:40].strip()
        feed[sym] = bars
    return feed
