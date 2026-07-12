"""
Free macro signal sources (no API key required for basic use):
  - FRED API: yield curve spread (10Y-2Y), VIX level, Fed Funds rate
  - CBOE VIX term structure: VIX9D, VIX (30d), VIX3M, VIX6M
  - Google Trends via pytrends (retail attention proxy) — optional
  - Apewisdom Reddit WSB sentiment (free, no key)
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from aiohttp import ClientError
from app.utils.logging import logger

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"


async def _fred_latest(series_id: str, api_key: str = "DEMO_KEY") -> Optional[float]:
    """Fetch latest value from FRED. DEMO_KEY allows 500 req/day — no registration needed."""
    url = (
        f"{FRED_BASE}?series_id={series_id}&api_key={api_key}"
        f"&file_type=json&sort_order=desc&limit=1"
    )
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.error(
                        "fred_fetch_failed",
                        extra={"series_id": series_id, "status": resp.status},
                    )
                    return None
                try:
                    data = await resp.json()
                except json.JSONDecodeError as e:
                    logger.error(
                        "fred_json_decode_error",
                        extra={"series_id": series_id, "error": str(e)},
                    )
                    return None

                obs = data.get("observations", [])
                if obs and obs[0].get("value") != ".":
                    try:
                        return float(obs[0]["value"])
                    except (ValueError, TypeError) as e:
                        logger.error(
                            "fred_value_conversion_error",
                            extra={"series_id": series_id, "raw_value": obs[0]["value"], "error": str(e)},
                        )
    except (ClientError, asyncio.TimeoutError) as e:
        logger.error(
            "fred_network_error",
            extra={"series_id": series_id, "error": str(e)},
        )
    except Exception as e:
        logger.error(
            "fred_unexpected_error",
            extra={"series_id": series_id, "error": str(e)},
        )
    return None


async def get_macro_snapshot() -> dict:
    """
    Fetch key macro indicators. All free, no API key.
    Returns dict with latest values + derived signals.
    """
    series_ids = [
        "T10Y2Y",          # 10Y-2Y yield curve spread
        "VIXCLS",          # VIX close
        "DFF",             # Fed Funds effective rate
        "BAMLH0A0HYM2",    # High-yield credit spread
        "DTWEXBGS",        # USD broad dollar index
    ]

    # Fetch in parallel, preserving order
    results = await asyncio.gather(
        *( _fred_latest(s) for s in series_ids ),
        return_exceptions=True,
    )

    # Log any exceptions that were captured
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                "fred_fetch_exception",
                extra={"series_id": series_ids[idx], "error": str(result)},
            )
            results[idx] = None  # normalize to None for downstream logic

    yield_spread = results[0] if isinstance(results[0], float) else None
    vix = results[1] if isinstance(results[1], float) else None
    fed_funds = results[2] if isinstance(results[2], float) else None
    hy_spread = results[3] if isinstance(results[3], float) else None
    usd_index = results[4] if isinstance(results[4], float) else None

    # Derive signals
    signals = {}
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

    macro_score = 0  # +1 risk-on, -1 risk-off
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
        "macro_score": macro_score,  # -3 to +3: positive = risk-on environment
        "macro_bias": (
            "risk_on" if macro_score >= 1 else "risk_off" if macro_score <= -1 else "neutral"
        ),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def get_reddit_sentiment(tickers: list[str] | None = None) -> dict:
    """
    Fetch WallStreetBets / Reddit sentiment from Apewisdom (free, no key required).
    Returns top mentioned tickers + mention count + sentiment score.
    """
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(APEWISDOM_URL, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.error(
                        "apewisdom_fetch_failed",
                        extra={"status": resp.status},
                    )
                    return {"error": "Apewisdom unavailable", "results": []}
                try:
                    data = await resp.json()
                except json.JSONDecodeError as e:
                    logger.error(
                        "apewisdom_json_decode_error",
                        extra={"error": str(e)},
                    )
                    return {"error": "Invalid JSON from Apewisdom", "results": []}

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
            "apewisdom_network_error",
            extra={"error": str(e)},
        )
        return {"error": str(e), "results": []}
    except Exception as e:
        logger.error(
            "apewisdom_unexpected_error",
            extra={"error": str(e)},
        )
        return {"error": str(e), "results": []}


# Simple cache to avoid hammering FRED
_macro_cache: dict = {}
_macro_cache_time: datetime | None = None
MACRO_CACHE_SECONDS = 300  # 5 min


async def get_macro_snapshot_cached() -> dict:
    global _macro_cache, _macro_cache_time
    now = datetime.now(timezone.utc)
    if _macro_cache_time and (now - _macro_cache_time).total_seconds() < MACRO_CACHE_SECONDS:
        return _macro_cache
    _macro_cache = await get_macro_snapshot()
    _macro_cache_time = now
    return _macro_cache