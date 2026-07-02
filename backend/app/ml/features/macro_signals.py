"""
Free macro signal sources (no API key required for basic use):
  - FRED API: yield curve spread (10Y-2Y), VIX level, Fed Funds rate
  - CBOE VIX term structure: VIX9D, VIX (30d), VIX3M, VIX6M
  - Google Trends via pytrends (retail attention proxy) — optional
  - Apewisdom Reddit WSB sentiment (free, no key)
"""
from __future__ import annotations

import asyncio
import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientConnectorError
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from app.utils.logging import logger

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"


async def _fred_latest(series_id: str, api_key: str = "DEMO_KEY") -> Optional[float]:
    """Fetch the latest numeric value for a given FRED series.

    Returns ``None`` if the request fails, the response is malformed, or the value is not
    a valid float.
    """
    url = f"{FRED_BASE}?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
    timeout = aiohttp.ClientTimeout(total=5)

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.error(
                        "FRED request failed",
                        extra={"series_id": series_id, "status": resp.status, "url": url},
                    )
                    return None

                try:
                    data = await resp.json()
                except (aiohttp.ContentTypeError, ValueError) as e:
                    logger.error(
                        "Failed to decode FRED JSON response",
                        extra={"series_id": series_id, "url": url, "error": str(e)},
                    )
                    return None

                observations = data.get("observations", [])
                if not observations:
                    logger.debug(
                        "No observations returned from FRED",
                        extra={"series_id": series_id, "url": url},
                    )
                    return None

                value_str = observations[0].get("value", ".")
                if value_str == ".":
                    logger.debug(
                        "FRED observation contains placeholder value",
                        extra={"series_id": series_id, "url": url},
                    )
                    return None

                return float(value_str)

    except (ClientError, asyncio.TimeoutError) as e:
        logger.error(
            "Network error while fetching FRED series",
            extra={"series_id": series_id, "url": url, "error": str(e)},
        )
    except Exception as e:  # Catch‑all for unexpected errors
        logger.exception(
            "Unexpected error in _fred_latest",
            extra={"series_id": series_id, "url": url},
        )
    return None


async def get_macro_snapshot() -> Dict[str, Any]:
    """
    Fetch key macro indicators. All free, no API key.
    Returns a dictionary with the latest values and derived signals.
    """
    # Fetch in parallel
    results = await asyncio.gather(
        _fred_latest("T10Y2Y"),       # 10Y-2Y yield curve spread (negative = inverted = recession risk)
        _fred_latest("VIXCLS"),       # VIX close (CBOE Volatility Index)
        _fred_latest("DFF"),          # Fed Funds effective rate
        _fred_latest("BAMLH0A0HYM2"), # High‑yield credit spread (recession proxy)
        _fred_latest("DTWEXBGS"),     # USD broad dollar index
        return_exceptions=True,
    )

    # Log any exceptions returned by asyncio.gather
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            metric = ["yield_spread", "vix", "fed_funds", "hy_spread", "usd_index"][idx]
            logger.error(
                "Error fetching macro metric",
                extra={"metric": metric, "exception": str(result)},
            )

    yield_spread = results[0] if isinstance(results[0], float) else None
    vix = results[1] if isinstance(results[1], float) else None
    fed_funds = results[2] if isinstance(results[2], float) else None
    hy_spread = results[3] if isinstance(results[3], float) else None
    usd_index = results[4] if isinstance(results[4], float) else None

    # Derive signals
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
        signals["vix_regime"] = "fear" if vix > 30 else "elevated" if vix > 20 else "complacent"
        signals["vix_level"] = vix

    if hy_spread is not None:
        signals["credit_stress"] = hy_spread > 5.0  # > 500bps = stress
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
        "macro_bias": "risk_on"
        if macro_score >= 1
        else "risk_off"
        if macro_score <= -1
        else "neutral",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def get_reddit_sentiment(tickers: List[str] | None = None) -> Dict[str, Any]:
    """
    Fetch WallStreetBets / Reddit sentiment from Apewisdom (free, no key required).
    Returns top mentioned tickers, mention count, and sentiment score.
    """
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(APEWISDOM_URL, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.error(
                        "Apewisdom request failed",
                        extra={"status": resp.status, "url": APEWISDOM_URL},
                    )
                    return {"error": "Apewisdom unavailable", "results": []}

                try:
                    data = await resp.json()
                except (aiohttp.ContentTypeError, ValueError) as e:
                    logger.error(
                        "Failed to decode Apewisdom JSON response",
                        extra={"url": APEWISDOM_URL, "error": str(e)},
                    )
                    return {"error": "invalid response format", "results": []}

                results = data.get("results", [])
                # Filter to requested tickers if specified
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

    except (ClientError, asyncio.TimeoutError) as e:
        logger.error(
            "Network error while fetching Apewisdom sentiment",
            extra={"url": APEWISDOM_URL, "error": str(e)},
        )
        return {"error": str(e), "results": []}
    except Exception as e:
        logger.exception(
            "Unexpected error in get_reddit_sentiment",
            extra={"url": APEWISDOM_URL},
        )
        return {"error": str(e), "results": []}


# Simple cache to avoid hammering FRED
_macro_cache: Dict[str, Any] = {}
_macro_cache_time: datetime | None = None
MACRO_CACHE_SECONDS = 300  # 5 min


async def get_macro_snapshot_cached() -> Dict[str, Any]:
    """Return a cached macro snapshot if recent; otherwise fetch a fresh one."""
    global _macro_cache, _macro_cache_time
    now = datetime.now(timezone.utc)

    if _macro_cache_time and (now - _macro_cache_time).total_seconds() < MACRO_CACHE_SECONDS:
        logger.debug("Returning cached macro snapshot", extra={"age_seconds": (now - _macro_cache_time).total_seconds()})
        return _macro_cache

    try:
        _macro_cache = await get_macro_snapshot()
        _macro_cache_time = now
    except Exception as e:
        logger.exception("Failed to refresh macro snapshot cache")
        # Preserve previous cache if it exists; otherwise propagate empty dict
        if not _macro_cache:
            _macro_cache = {}
        _macro_cache_time = now  # avoid rapid retry loops

    return _macro_cache