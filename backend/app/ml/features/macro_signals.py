"""
Free macro signal sources (no API key required for basic use):
  - FRED API: yield curve spread (10Y-2Y), VIX level, Fed Funds rate
  - CBOE VIX term structure: VIX9D, VIX (30d), VIX3M, VIX6M
  - Google Trends via pytrends (retail attention proxy) — optional
  - Apewisdom Reddit WSB sentiment (free, no key)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple

import aiohttp
from app.utils.logging import logger

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"


async def _fred_fetch(series_id: str, limit: int = 1, api_key: str = "DEMO_KEY") -> Optional[List[Tuple[datetime, float]]]:
    """Fetch the most recent `limit` observations for a FRED series."""
    url = (
        f"{FRED_BASE}?series_id={series_id}"
        f"&api_key={api_key}&file_type=json&sort_order=desc&limit={limit}"
    )
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.debug(f"FRED fetch status {resp.status} for {series_id}")
                    return None
                data = await resp.json()
                obs = data.get("observations", [])
                result: List[Tuple[datetime, float]] = []
                for entry in obs:
                    val = entry.get("value")
                    if val is None or val == ".":
                        continue
                    dt_str = entry.get("date")
                    if not dt_str:
                        continue
                    try:
                        dt = datetime.strptime(dt_str, "%Y-%m-%d")
                    except ValueError:
                        continue
                    result.append((dt, float(val)))
                return result if result else None
    except Exception as e:
        logger.debug(f"FRED fetch {series_id}: {e}")
    return None


async def _fred_latest(series_id: str, api_key: str = "DEMO_KEY") -> Optional[float]:
    """Convenience wrapper returning the most recent value."""
    data = await _fred_fetch(series_id, limit=1, api_key=api_key)
    return data[0][1] if data else None


def _is_decreasing(trend: List[Tuple[datetime, float]]) -> bool:
    """Return True if the series shows a decreasing trend over the supplied points."""
    if len(trend) < 2:
        return False
    # Simple check: last value lower than first value
    return trend[-1][1] < trend[0][1]


async def get_macro_snapshot() -> dict:
    """
    Fetch key macro indicators. All free, no API key.
    Returns dict with latest values + derived signals and entry/exit flags.
    """
    # Parallel fetch of latest values
    latest_fut = asyncio.gather(
        _fred_latest("T10Y2Y"),       # 10Y-2Y yield curve spread (negative = inverted = recession risk)
        _fred_latest("VIXCLS"),       # VIX close (CBOE Volatility Index)
        _fred_latest("DFF"),          # Fed Funds effective rate
        _fred_latest("BAMLH0A0HYM2"), # High‑yield credit spread (recession proxy)
        _fred_latest("DTWEXBGS"),     # USD broad dollar index
        return_exceptions=True,
    )
    # Fetch recent VIX trend (last 2 observations)
    vix_trend_fut = _fred_fetch("VIXCLS", limit=2)

    latest_results = await latest_fut
    vix_trend_data = await vix_trend_fut

    yield_spread = latest_results[0] if isinstance(latest_results[0], float) else None
    vix = latest_results[1] if isinstance(latest_results[1], float) else None
    fed_funds = latest_results[2] if isinstance(latest_results[2], float) else None
    hy_spread = latest_results[3] if isinstance(latest_results[3], float) else None
    usd_index = latest_results[4] if isinstance(latest_results[4], float) else None

    # Derived signals
    signals: dict = {}
    if yield_spread is not None:
        signals["yield_curve_inverted"] = yield_spread < 0
        signals["yield_spread_bps"] = round(yield_spread * 100, 1)
        signals["yield_curve_signal"] = (
            "risk_off" if yield_spread < -0.5 else "neutral" if yield_spread < 0.5 else "risk_on"
        )

    if vix is not None:
        signals["vix_regime"] = "fear" if vix > 30 else "elevated" if vix > 20 else "complacent"
        signals["vix_level"] = vix

    if hy_spread is not None:
        signals["credit_stress"] = hy_spread > 5.0  # > 500bps = stress
        signals["hy_spread_pct"] = hy_spread

    # Macro score – tighter weighting: require consensus of at least two bullish signals for +1
    macro_score = 0
    bullish = 0
    bearish = 0

    if yield_spread is not None:
        if yield_spread > 0.5:
            bullish += 1
        elif yield_spread < -0.5:
            bearish += 1

    if vix is not None:
        if vix < 15:
            bullish += 1
        elif vix > 30:
            bearish += 1

    if hy_spread is not None:
        if hy_spread < 3.0:
            bullish += 1
        elif hy_spread > 6.0:
            bearish += 1

    macro_score = bullish - bearish

    # Confirmation filters
    entry_signal = (
        macro_score >= 2
        and vix is not None
        and vix < 20
        and (vix_trend_data is None or _is_decreasing(vix_trend_data))
    )

    exit_signal = macro_score <= -1 or (vix is not None and vix > 30)

    return {
        "yield_spread_10y2y": yield_spread,
        "vix": vix,
        "fed_funds_rate": fed_funds,
        "hy_credit_spread": hy_spread,
        "usd_index": usd_index,
        "signals": signals,
        "macro_score": macro_score,  # -3 to +3: positive = risk‑on environment
        "macro_bias": "risk_on" if macro_score >= 1 else "risk_off" if macro_score <= -1 else "neutral",
        "entry_signal": entry_signal,
        "exit_signal": exit_signal,
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
                    return {"error": "Apewisdom unavailable", "results": []}
                data = await resp.json()
                results = data.get("results", [])
                # Filter to requested tickers if specified
                if tickers:
                    ticker_set = {t.upper() for t in tickers}
                    results = [r for r in results if r.get("ticker", "").upper() in ticker_set]
                return {
                    "results": results[:20],
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "source": "apewisdom.io (reddit wsb)",
                }
    except Exception as e:
        logger.debug(f"Apewisdom fetch error: {e}")
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