"""
Macro signal utilities.

Provides free macroeconomic indicator snapshots and Reddit (WallStreetBets) sentiment
without requiring API keys. Functions are asynchronous and can be cached to reduce
load on external services.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, date, timedelta
from typing import Any, Optional, List, Dict

import aiohttp
from app.utils.logging import logger

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"


async def _fred_latest(series_id: str, api_key: str = "DEMO_KEY") -> Optional[float]:
    """
    Retrieve the most recent observation for a FRED series.

    Parameters
    ----------
    series_id: str
        Identifier of the FRED series (e.g., ``T10Y2Y``).
    api_key: str, optional
        API key for FRED. The demo key permits up to 500 requests per day.

    Returns
    -------
    Optional[float]
        The latest numeric value, or ``None`` if the request fails or the value is
        missing/invalid.
    """
    url = (
        f"{FRED_BASE}?series_id={series_id}"
        f"&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
    )
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


async def get_macro_snapshot() -> Dict[str, Any]:
    """
    Fetch a snapshot of key macroeconomic indicators.

    Returns a dictionary containing raw indicator values, derived signals, a
    composite macro score, and a timestamp of when the data were fetched.

    The indicators are obtained from free public endpoints (FRED and others) and
    include:

    * 10‑year/2‑year yield curve spread
    * VIX level
    * Effective Fed Funds rate
    * High‑yield credit spread
    * Broad USD index

    Derived signals translate these raw numbers into qualitative regimes such as
    ``risk_on`` or ``risk_off``.
    """
    results = await asyncio.gather(
        _fred_latest("T10Y2Y"),       # 10Y-2Y yield curve spread
        _fred_latest("VIXCLS"),       # VIX close
        _fred_latest("DFF"),          # Fed Funds effective rate
        _fred_latest("BAMLH0A0HYM2"), # High‑yield credit spread
        _fred_latest("DTWEXBGS"),     # USD broad dollar index
        return_exceptions=True,
    )

    yield_spread = results[0] if isinstance(results[0], float) else None
    vix = results[1] if isinstance(results[1], float) else None
    fed_funds = results[2] if isinstance(results[2], float) else None
    hy_spread = results[3] if isinstance(results[3], float) else None
    usd_index = results[4] if isinstance(results[4], float) else None

    signals: Dict[str, Any] = {}
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
            "fear" if vix > 30 else "elevated" if vix > 20 else "complacent"
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
        macro_score += (
            1 if hy_spread < 3.5 else -1 if hy_spread > 6.0 else 0
        )

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


async def get_reddit_sentiment(tickers: List[str] | None = None) -> Dict[str, Any]:
    """
    Retrieve recent WallStreetBets sentiment from Apewisdom.

    Parameters
    ----------
    tickers: list[str] | None, optional
        If provided, filter results to only include these ticker symbols
        (case‑insensitive). When ``None``, all tickers are returned.

    Returns
    -------
    dict
        A dictionary containing up to 20 result entries, the fetch timestamp,
        and the data source. If an error occurs, an ``error`` key is added.
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
                        r
                        for r in results
                        if r.get("ticker", "").upper() in ticker_set
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
_macro_cache: Dict[str, Any] = {}
_macro_cache_time: Optional[datetime] = None
MACRO_CACHE_SECONDS = 300  # 5 min


async def get_macro_snapshot_cached() -> Dict[str, Any]:
    """
    Return a cached macro snapshot if it is still fresh.

    The cache expires after :data:`MACRO_CACHE_SECONDS`. When expired, a fresh
    snapshot is fetched via :func:`get_macro_snapshot` and stored.
    """
    global _macro_cache, _macro_cache_time
    now = datetime.now(timezone.utc)
    if (
        _macro_cache_time
        and (now - _macro_cache_time).total_seconds() < MACRO_CACHE_SECONDS
    ):
        return _macro_cache
    _macro_cache = await get_macro_snapshot()
    _macro_cache_time = now
    return _macro_cache