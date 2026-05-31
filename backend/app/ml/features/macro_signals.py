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
from datetime import datetime, timezone, date, timedelta
from typing import Optional
from app.utils.logging import logger


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"


async def _fred_latest(series_id: str, api_key: str = "DEMO_KEY") -> Optional[float]:
    """Fetch latest value from FRED. DEMO_KEY allows 500 req/day — no registration needed."""
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


async def get_macro_snapshot() -> dict:
    """
    Fetch key macro indicators. All free, no API key.
    Returns dict with latest values + derived signals.
    """
    # Fetch in parallel
    results = await asyncio.gather(
        _fred_latest("T10Y2Y"),       # 10Y-2Y yield curve spread (negative = inverted = recession risk)
        _fred_latest("VIXCLS"),       # VIX close (CBOE Volatility Index)
        _fred_latest("DFF"),          # Fed Funds effective rate
        _fred_latest("BAMLH0A0HYM2"), # High-yield credit spread (recession proxy)
        _fred_latest("DTWEXBGS"),     # USD broad dollar index
        return_exceptions=True,
    )

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
        signals["yield_curve_signal"] = "risk_off" if yield_spread < -0.5 else "neutral" if yield_spread < 0.5 else "risk_on"

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
        "macro_score": macro_score,           # -3 to +3: positive = risk-on environment
        "macro_bias": "risk_on" if macro_score >= 1 else "risk_off" if macro_score <= -1 else "neutral",
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


def _yf_last_close(ticker: str) -> Optional[float]:
    """
    Sync helper: fetch the most recent close for a ticker via yfinance.
    Runs inside asyncio.to_thread (yfinance is blocking). Returns None on any failure.
    Never raises, never fabricates a value.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            return None
        val = float(closes.iloc[-1])
        # Guard against NaN / non-finite
        if val != val or val in (float("inf"), float("-inf")):
            return None
        return val
    except Exception as e:
        logger.debug(f"yfinance fetch {ticker}: {e}")
        return None


async def _yf_latest(ticker: str) -> Optional[float]:
    """Async wrapper around _yf_last_close using asyncio.to_thread."""
    try:
        return await asyncio.to_thread(_yf_last_close, ticker)
    except Exception as e:
        logger.debug(f"yfinance thread {ticker}: {e}")
        return None


async def get_index_feeds() -> dict:
    """
    Extended macro / index data feeds — all FREE (FRED DEMO_KEY + yfinance fallback).

    Covers jobs market, gold, oil, dollar (DXY), VIX term structure (VIX9D/VIX),
    NYSE TICK breadth, and a leading-vs-lagging composite with a leading-indicator
    score. Every feed degrades gracefully: any source that fails contributes None,
    and the function never raises and never fabricates data.

    Returns a dict shaped like get_macro_snapshot():
      individual values + a "signals" sub-dict + "leading_score" (-3..+3) + "fetched_at".

    FRED series used:
      ICSA           initial jobless claims (leading, weekly)
      UNRATE         unemployment rate (lagging, monthly)
      PAYEMS         total nonfarm payrolls (lagging, monthly)
      GOLDAMGBD228NLBM  London gold fixing (gold fallback)
      DCOILWTICO     WTI crude oil spot (oil fallback)
      T10Y2Y         10Y-2Y yield curve spread (leading)
      BAMLH0A0HYM2   high-yield credit spread (leading)
    yfinance tickers used:
      GC=F (gold), CL=F (WTI oil), DX-Y.NYB (DXY dollar index),
      ^VIX9D (9-day VIX), ^VIX (30-day VIX), ^TICK (NYSE TICK breadth)
    """
    # --- FRED fetches (parallel) ---
    fred = await asyncio.gather(
        _fred_latest("ICSA"),         # initial jobless claims (leading)
        _fred_latest("UNRATE"),       # unemployment rate (lagging)
        _fred_latest("PAYEMS"),       # nonfarm payrolls (lagging)
        _fred_latest("GOLDAMGBD228NLBM"),  # London gold fix (gold fallback)
        _fred_latest("DCOILWTICO"),   # WTI crude (oil fallback)
        _fred_latest("T10Y2Y"),       # yield curve (leading)
        _fred_latest("BAMLH0A0HYM2"), # HY credit spread (leading)
        return_exceptions=True,
    )

    def _f(idx):
        v = fred[idx]
        return v if isinstance(v, float) else None

    jobless_claims = _f(0)
    unemployment = _f(1)
    payrolls = _f(2)
    gold_fred = _f(3)
    oil_fred = _f(4)
    yield_spread = _f(5)
    hy_spread = _f(6)

    # --- yfinance fetches (parallel; each returns None on failure) ---
    yf_results = await asyncio.gather(
        _yf_latest("GC=F"),       # gold front-month future
        _yf_latest("CL=F"),       # WTI crude future
        _yf_latest("DX-Y.NYB"),   # DXY dollar index
        _yf_latest("^VIX9D"),     # 9-day VIX
        _yf_latest("^VIX"),       # 30-day VIX
        _yf_latest("^TICK"),      # NYSE TICK (may be unavailable)
        return_exceptions=True,
    )

    def _y(idx):
        v = yf_results[idx]
        return v if isinstance(v, float) else None

    gold_yf = _y(0)
    oil_yf = _y(1)
    dxy = _y(2)
    vix9d = _y(3)
    vix_30d = _y(4)
    tick = _y(5)

    # Prefer real-time yfinance price, fall back to FRED series
    gold = gold_yf if gold_yf is not None else gold_fred
    oil = oil_yf if oil_yf is not None else oil_fred

    # --- Derived signals ---
    signals: dict = {}

    # Jobs market
    if jobless_claims is not None:
        signals["initial_jobless_claims"] = jobless_claims
    if unemployment is not None:
        signals["unemployment_rate"] = unemployment
    if payrolls is not None:
        signals["nonfarm_payrolls_k"] = payrolls

    # VIX term structure (VIX9D / VIX). Ratio < 1 => backwardation => acute near-term fear.
    vix_term_ratio = None
    if vix9d is not None and vix_30d is not None and vix_30d > 0:
        vix_term_ratio = round(vix9d / vix_30d, 4)
        signals["vix_term_ratio"] = vix_term_ratio
        if vix_term_ratio < 1.0:
            signals["vix_term_structure"] = "backwardation_acute_fear"
        elif vix_term_ratio > 1.1:
            signals["vix_term_structure"] = "steep_contango_complacent"
        else:
            signals["vix_term_structure"] = "normal_contango"

    # NYSE TICK breadth
    if tick is not None:
        signals["nyse_tick"] = tick
        signals["tick_breadth"] = "strong_buying" if tick > 600 else "strong_selling" if tick < -600 else "neutral"

    # Dollar
    if dxy is not None:
        signals["dxy"] = dxy

    # --- Leading vs lagging composite ---
    composite = {
        "leading": {
            "initial_jobless_claims": jobless_claims,
            "yield_spread_10y2y": yield_spread,
            "hy_credit_spread": hy_spread,
            "vix9d": vix9d,
        },
        "coincident": {
            "wti_oil": oil,
            "dxy": dxy,
            "nyse_tick": tick,
        },
        "lagging": {
            "unemployment_rate": unemployment,
            "nonfarm_payrolls": payrolls,
            "gold": gold,
        },
    }

    # --- Leading-indicator score (-3..+3): positive = improving, negative = deteriorating ---
    # Deterioration drivers: rising jobless claims (proxied by elevated level),
    # inverted curve, widening credit spreads.
    leading_score = 0
    if jobless_claims is not None:
        # No history available from a single point; use absolute stress threshold.
        # ~300k+ initial claims historically signals labor-market softening.
        leading_score += -1 if jobless_claims > 300_000 else 1
    if yield_spread is not None:
        leading_score += -1 if yield_spread < 0 else 1
    if hy_spread is not None:
        leading_score += -1 if hy_spread > 5.0 else 1 if hy_spread < 3.5 else 0
    # Clamp to -3..+3
    leading_score = max(-3, min(3, leading_score))

    signals["leading_regime"] = (
        "deteriorating" if leading_score <= -1
        else "improving" if leading_score >= 1
        else "neutral"
    )

    return {
        # Jobs market
        "initial_jobless_claims": jobless_claims,
        "unemployment_rate": unemployment,
        "nonfarm_payrolls": payrolls,
        # Commodities
        "gold": gold,
        "oil_wti": oil,
        # Dollar
        "dxy": dxy,
        # Fear / term structure
        "vix9d": vix9d,
        "vix_30d": vix_30d,
        "vix_term_ratio": vix_term_ratio,
        # Breadth
        "nyse_tick": tick,
        # Composite
        "composite": composite,
        "signals": signals,
        "leading_score": leading_score,   # -3..+3: positive = improving, negative = deteriorating
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


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


# Simple cache for the extended index feeds (10 min TTL)
_index_cache: dict = {}
_index_cache_time: datetime | None = None
INDEX_CACHE_SECONDS = 600  # 10 min


async def get_index_feeds_cached() -> dict:
    global _index_cache, _index_cache_time
    now = datetime.now(timezone.utc)
    if _index_cache_time and (now - _index_cache_time).total_seconds() < INDEX_CACHE_SECONDS:
        return _index_cache
    _index_cache = await get_index_feeds()
    _index_cache_time = now
    return _index_cache
