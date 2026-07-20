"""
Free macro signal sources (no API key required for basic use):
  - FRED API: yield curve spread (10Y-2Y), VIX level, Fed Funds rate
  - CBOE VIX term structure: VIX9D, VIX (30d), VIX3M, VIX6M
  - Google Trends via pytrends (retail attention proxy) — optional
  - Apewisdom Reddit WSB sentiment (free, no key)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel, Field, validator

from app.utils.logging import logger

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"


class MacroSignals(BaseModel):
    """Derived macro signals based on raw indicator values."""

    yield_curve_inverted: Optional[bool] = Field(
        None,
        description="True when the 10y‑2y spread is negative (inverted yield curve).",
        example=False,
    )
    yield_spread_bps: Optional[float] = Field(
        None,
        description="Yield spread expressed in basis points.",
        example=25.4,
    )
    yield_curve_signal: Optional[str] = Field(
        None,
        description="Qualitative signal derived from the yield spread.",
        example="neutral",
        pattern="^(risk_off|neutral|risk_on)$",
    )
    vix_regime: Optional[str] = Field(
        None,
        description="Volatility regime based on VIX level.",
        example="elevated",
        pattern="^(fear|elevated|complacent)$",
    )
    vix_level: Optional[float] = Field(
        None,
        description="Current VIX index level.",
        example=22.7,
    )
    credit_stress: Optional[bool] = Field(
        None,
        description="True when high‑yield credit spread exceeds stress threshold (5.0%).",
        example=False,
    )
    hy_spread_pct: Optional[float] = Field(
        None,
        description="High‑yield credit spread expressed as a percentage.",
        example=4.3,
    )


class MacroSnapshot(BaseModel):
    """Container for macro indicator values and derived signals."""

    yield_spread_10y2y: Optional[float] = Field(
        None,
        description="Latest 10‑year minus 2‑year Treasury yield spread (in percent).",
        example=-0.12,
    )
    vix: Optional[float] = Field(
        None,
        description="Current VIX index level.",
        example=18.3,
    )
    fed_funds_rate: Optional[float] = Field(
        None,
        description="Effective Federal Funds rate (percent).",
        example=5.25,
    )
    hy_credit_spread: Optional[float] = Field(
        None,
        description="High‑yield credit spread (percent).",
        example=4.8,
    )
    usd_index: Optional[float] = Field(
        None,
        description="Broad USD index value.",
        example=102.4,
    )
    signals: MacroSignals = Field(
        default_factory=MacroSignals,
        description="Derived macro signals.",
    )
    macro_score: int = Field(
        ...,
        description="Aggregated risk score ranging from -3 (risk‑off) to +3 (risk‑on).",
        example=1,
    )
    macro_bias: str = Field(
        ...,
        description="Overall macro bias derived from macro_score.",
        example="risk_on",
        pattern="^(risk_on|risk_off|neutral)$",
    )
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of when the snapshot was fetched (UTC).",
        example="2024-01-01T12:34:56.789Z",
    )

    @validator("macro_score")
    def score_range(cls, v: int) -> int:
        if not -3 <= v <= 3:
            raise ValueError("macro_score must be between -3 and 3")
        return v

    @validator("macro_bias")
    def bias_consistency(cls, v: str, values: Dict[str, Any]) -> str:
        score = values.get("macro_score")
        if score is None:
            return v
        expected = (
            "risk_on"
            if score >= 1
            else "risk_off"
            if score <= -1
            else "neutral"
        )
        if v != expected:
            raise ValueError(f"macro_bias '{v}' inconsistent with macro_score {score}")
        return v


class RedditResult(BaseModel):
    """Single Reddit sentiment entry."""

    ticker: str = Field(..., description="Ticker symbol.", example="AAPL")
    mentions: Optional[int] = Field(
        None,
        description="Number of times the ticker was mentioned.",
        example=152,
    )
    sentiment_score: Optional[float] = Field(
        None,
        description="Aggregate sentiment score for the ticker.",
        example=0.42,
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original payload from the Apewisdom API.",
    )


class RedditSentiment(BaseModel):
    """Aggregated Reddit sentiment response."""

    results: List[RedditResult] = Field(
        default_factory=list,
        description="Top Reddit sentiment entries (max 20).",
    )
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of fetch (UTC).",
        example="2024-01-01T12:34:56.789Z",
    )
    source: str = Field(
        "apewisdom.io (reddit wsb)",
        description="Data source identifier.",
        example="apewisdom.io (reddit wsb)",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if the request failed.",
        example="Apewisdom unavailable",
    )


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


async def get_macro_snapshot() -> MacroSnapshot:
    """Fetch key macro indicators and return a validated MacroSnapshot model."""
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

    signals = MacroSignals()
    if yield_spread is not None:
        signals.yield_curve_inverted = yield_spread < 0
        signals.yield_spread_bps = round(yield_spread * 100, 1)
        signals.yield_curve_signal = (
            "risk_off"
            if yield_spread < -0.5
            else "neutral"
            if yield_spread < 0.5
            else "risk_on"
        )

    if vix is not None:
        signals.vix_regime = "fear" if vix > 30 else "elevated" if vix > 20 else "complacent"
        signals.vix_level = vix

    if hy_spread is not None:
        signals.credit_stress = hy_spread > 5.0  # > 500bps = stress
        signals.hy_spread_pct = hy_spread

    macro_score = 0  # +1 risk-on, -1 risk-off
    if yield_spread is not None:
        macro_score += 1 if yield_spread > 0 else -1
    if vix is not None:
        macro_score += 1 if vix < 20 else -1 if vix > 30 else 0
    if hy_spread is not None:
        macro_score += 1 if hy_spread < 3.5 else -1 if hy_spread > 6.0 else 0

    macro_bias = (
        "risk_on"
        if macro_score >= 1
        else "risk_off"
        if macro_score <= -1
        else "neutral"
    )

    snapshot = MacroSnapshot(
        yield_spread_10y2y=yield_spread,
        vix=vix,
        fed_funds_rate=fed_funds,
        hy_credit_spread=hy_spread,
        usd_index=usd_index,
        signals=signals,
        macro_score=macro_score,
        macro_bias=macro_bias,
        fetched_at=datetime.now(timezone.utc),
    )
    return snapshot


async def get_reddit_sentiment(tickers: List[str] | None = None) -> RedditSentiment:
    """Fetch WallStreetBets / Reddit sentiment from Apewisdom (free, no key required)."""
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(APEWISDOM_URL, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return RedditSentiment(
                        results=[],
                        error="Apewisdom unavailable",
                        source="apewisdom.io (reddit wsb)",
                    )
                data = await resp.json()
                raw_results = data.get("results", [])
                if tickers:
                    ticker_set = {t.upper() for t in tickers}
                    raw_results = [
                        r for r in raw_results if r.get("ticker", "").upper() in ticker_set
                    ]
                parsed_results = [
                    RedditResult(
                        ticker=r.get("ticker", ""),
                        mentions=r.get("mentions"),
                        sentiment_score=r.get("sentiment_score"),
                        raw=r,
                    )
                    for r in raw_results[:20]
                ]
                return RedditSentiment(
                    results=parsed_results,
                    source="apewisdom.io (reddit wsb)",
                )
    except Exception as e:
        logger.debug(f"Apewisdom fetch error: {e}")
        return RedditSentiment(
            results=[],
            error=str(e),
            source="apewisdom.io (reddit wsb)",
        )


# Simple cache to avoid hammering FRED
_macro_cache: dict = {}
_macro_cache_time: Optional[datetime] = None
MACRO_CACHE_SECONDS = 300  # 5 min


async def get_macro_snapshot_cached() -> MacroSnapshot:
    """Return a cached MacroSnapshot if recent; otherwise fetch a fresh snapshot."""
    global _macro_cache, _macro_cache_time
    now = datetime.now(timezone.utc)
    if _macro_cache_time and (now - _macro_cache_time).total_seconds() < MACRO_CACHE_SECONDS:
        return MacroSnapshot.parse_obj(_macro_cache)
    snapshot = await get_macro_snapshot()
    _macro_cache = snapshot.dict()
    _macro_cache_time = now
    return snapshot

__all__ = [
    "MacroSnapshot",
    "MacroSignals",
    "RedditSentiment",
    "RedditResult",
    "get_macro_snapshot",
    "get_macro_snapshot_cached",
    "get_reddit_sentiment",
]