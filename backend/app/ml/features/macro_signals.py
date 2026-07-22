"""
Macro signal utilities for retrieving free macroeconomic indicators and Reddit sentiment.

This module provides asynchronous functions to fetch:
- Yield curve spread, VIX level, Fed Funds rate, high‑yield credit spread, and USD index
  from the Federal Reserve Economic Data (FRED) API (no API key required for the demo key).
- WallStreetBets sentiment from the Apewisdom public endpoint.

All functions are designed to be non‑blocking and return plain Python data
structures suitable for downstream feature engineering pipelines.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, date, timedelta
from typing import Any, List, Optional

import aiohttp
from app.utils.logging import logger

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"

async def _fred_latest(series_id: str, api_key: str = "DEMO_KEY") -> Optional[float]:
    """
    Retrieve the most recent observation for a given FRED series.

    Parameters
    ----------
    series_id: str
        The identifier of the FRED series (e.g., ``"T10Y2Y"`` for the 10‑year/2‑year yield spread).
    api_key: str, optional
        API key for FRED. ``"DEMO_KEY"`` provides up to 500 requests per day without registration.

    Returns
    -------
    Optional[float]
        The latest numeric value for the series, or ``None`` if the request fails,
        the response is malformed, or the value is missing (represented by a ``"."`` in
        the FRED response).
    """
    url = f"{FRED_BASE}?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                obs = data.get("observations", [])
                if obs and obs[0]["value"] != ".":
                    return float(obs[0]["value"])
    except Exception as e:
        logger.debug(f"FRED fetch {series_id}: {e}")
    return None


async def get_macro_snapshot() -> dict[str, Any]:
    """
    Fetch a snapshot of key macroeconomic indicators.

    The function queries several FRED series in parallel and derives simple
    binary and categorical signals used by the trading strategies.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the raw values, derived signals, an overall
        macro score (``-3`` to ``+3``), a textual macro bias, and the fetch timestamp.
    """
    results = await asyncio.gather(
        _fred_latest("T10Y2Y"),       # 10Y‑2Y yield curve spread (negative = inverted = recession risk)
        _fred_latest("VIXCLS"),       # VIX close (CBOE Volatility Index)
        _fred_latest("DFF"),          # Fed Funds effective rate
        _fred_latest("BAMLH0A0HYM2"), # High‑yield credit spread (recession proxy)
        _fred_latest("DTWEXBGS"),     # USD broad dollar index
        return_exceptions=True,
    )

    yield_spread = results[0] if isinstance(results[0], float) else None
    vix = results[1] if isinstance(results[1], float) else None
    fed_funds = results[2] if isinstance(results[2], float) else None
    hy_spread = results[3] if isinstance(results[3], float) else None
    usd_index = results[4] if isinstance(results[4], float) else None

    signals: dict[str, Any] = {}
    if yield_spread is not None:
        signals["yield_curve_inverted"] = yield_spread < 0
        signals["yield_spread_bps"] = round(yield_spread * 100, 1)
        signals["yield_curve_signal"] = (
            "risk_off"
            if yield_spread < -0.5
            else "neutral"
            if yield_spread < 0.5
            else "risk_on"
        )

    if vix is not None:
        signals["vix_regime"] = (
            "fear"
            if vix > 30
            else "elevated"
            if vix > 20
            else "complacent"
        )
        signals["vix_level"] = vix

    if hy_spread is not None:
        signals["credit_stress"] = hy_spread > 5.0  # > 500 bps = stress
        signals["hy_spread_pct"] = hy_spread

    macro_score = 0  # +1 risk‑on, -1 risk‑off
    if yield_spread is not None:
        macro_score += 1 if yield_spread > 0 else -1
    if vix is not None:
        macro_score += 1 if vix < 20 else -1 if vix > 30 else 0
    if hy_spread is not None:
        macro_score += 1 if hy_spread < 3.5 else -1 if hy_spread > 6.0 else 0

    return {
        "yield_spread_10y2y": yield_spread,
        "vix": vix,
        "fed_funds_rate": fed_funds,
        "hy_credit_spread": hy_spread,
        "usd_index": usd_index,
        "signals": signals,
        "macro_score": macro_score,  # -3 to +3: positive = risk‑on environment
        "macro_bias": (
            "risk_on"
            if macro_score >= 1
            else "risk_off"
            if macro_score <= -1
            else "neutral"
        ),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def get_reddit_sentiment(tickers: List[str] | None = None) -> dict[str, Any]:
    """
    Retrieve recent WallStreetBets sentiment from the public Apewisdom endpoint.

    Parameters
    ----------
    tickers: list[str] | None, optional
        If provided, the result set is filtered to include only the specified tickers
        (case‑insensitive). If ``None``, all tickers returned by the endpoint are considered.

    Returns
    -------
    dict[str, Any]
        A dictionary containing up to 20 result entries, the fetch timestamp, and a
        ``source`` identifier. In case of an error, an ``"error"`` key is included.
    """
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(APEWISDOM_URL, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return {"error": "Apewisdom unavailable", "results": []}
                data = await resp.json()
                results = data.get("results", [])
                if tickers:
                    ticker_set = {t.upper() for t in tickers}
                    results = [
                        r for r in results if r.get("ticker", "").upper() in ticker_set
                    ]
                return {
                    "results": results[:20],
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "source": "apewisdom.io (reddit wsb)",
                }
    except Exception as e:
        logger.debug(f"Apewisdom fetch error: {e}")
        return {"error": str(e), "results": []}


# Simple cache to avoid hammering FRED
_macro_cache: dict[str, Any] = {}
_macro_cache_time: datetime | None = None
MACRO_CACHE_SECONDS = 300  # 5 min


async def get_macro_snapshot_cached() -> dict[str, Any]:
    """
    Return a cached macro snapshot if it is younger than ``MACRO_CACHE_SECONDS``.

    The first call (or any call after the cache expires) invokes
    :func:`get_macro_snapshot` and stores its result for subsequent fast retrievals.

    Returns
    -------
    dict[str, Any]
        The same structure as produced by :func:`get_macro_snapshot`.
    """
    global _macro_cache, _macro_cache_time
    now = datetime.now(timezone.utc)
    if _macro_cache_time and (now - _macro_cache_time).total_seconds() < MACRO_CACHE_SECONDS:
        return _macro_cache
    _macro_cache = await get_macro_snapshot()
    _macro_cache_time = now
    return _macro_cache