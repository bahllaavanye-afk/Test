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
from typing import List, Optional

import aiohttp
from pydantic import BaseModel, Field, validator

from app.utils.logging import logger

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"


class MacroSignals(BaseModel):
    """Derived macro signals from raw indicator values."""

    yield_curve_inverted: Optional[bool] = Field(
        default=None,
        description="True if the 10Y‑2Y yield spread is negative (inverted).",
        example=True,
    )
    yield_spread_bps: Optional[float] = Field(
        default=None,
        description="Yield spread expressed in basis points.",
        example=-25.3,
    )
    yield_curve_signal: Optional[str] = Field(
        default=None,
        description="High‑level regime derived from the yield spread.",
        example="risk_off",
    )
    vix_regime: Optional[str] = Field(
        default=None,
        description="Risk regime based on VIX level.",
        example="elevated",
    )
    vix_level: Optional[float] = Field(
        default=None,
        description="Current VIX index level.",
        ge=0,
        example=22.5,
    )
    credit_stress: Optional[bool] = Field(
        default=None,
        description="True if high‑yield credit spread exceeds stress threshold.",
        example=False,
    )
    hy_spread_pct: Optional[float] = Field(
        default=None,
        description="High‑yield credit spread expressed as a percentage.",
        ge=0,
        example=4.2,
    )

    @validator("yield_curve_signal")
    def _validate_yield_curve_signal(cls, v):
        allowed = {"risk_off", "neutral", "risk_on"}
        if v is not None and v not in allowed:
            raise ValueError(f"yield_curve_signal must be one of {allowed}")
        return v

    @validator("vix_regime")
    def _validate_vix_regime(cls, v):
        allowed = {"fear", "elevated", "complacent"}
        if v is not None and v not in allowed:
            raise ValueError(f"vix_regime must be one of {allowed}")
        return v


class MacroSnapshot(BaseModel):
    """Aggregated macro indicator snapshot with derived signals."""

    yield_spread_10y2y: Optional[float] = Field(
        default=None,
        description="Latest 10‑year minus 2‑year Treasury yield spread (in %).",
        example=-0.45,
    )
    vix: Optional[float] = Field(
        default=None,
        description="Current VIX index level.",
        ge=0,
        example=21.3,
    )
    fed_funds_rate: Optional[float] = Field(
        default=None,
        description="Effective Federal Funds rate (in %).",
        ge=0,
        example=5.25,
    )
    hy_credit_spread: Optional[float] = Field(
        default=None,
        description="High‑yield credit spread (in %).",
        ge=0,
        example=4.8,
    )
    usd_index: Optional[float] = Field(
        default=None,
        description="Broad US Dollar index value.",
        example=102.5,
    )
    signals: MacroSignals = Field(
        default_factory=MacroSignals,
        description="Derived macro signals based on raw values.",
    )
    macro_score: int = Field(
        description="Aggregated macro risk score ranging from -3 (risk‑off) to +3 (risk‑on).",
        ge=-3,
        le=3,
        example=1,
    )
    macro_bias: str = Field(
        description="High‑level macro bias derived from macro_score.",
        example="risk_on",
    )
    fetched_at: datetime = Field(
        description="Timestamp of when the snapshot was retrieved (UTC).",
        example="2024-01-01T12:00:00Z",
    )

    @validator("macro_bias")
    def _validate_macro_bias(cls, v):
        allowed = {"risk_on", "risk_off", "neutral"}
        if v not in allowed:
            raise ValueError(f"macro_bias must be one of {allowed}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "yield_spread_10y2y": -0.45,
                "vix": 21.3,
                "fed_funds_rate": 5.25,
                "hy_credit_spread": 4.8,
                "usd_index": 102.5,
                "signals": {
                    "yield_curve_inverted": True,
                    "yield_spread_bps": -45.0,
                    "yield_curve_signal": "risk_off",
                    "vix_regime": "elevated",
                    "vix_level": 21.3,
                    "credit_stress": False,
                    "hy_spread_pct": 4.8,
                },
                "macro_score": 1,
                "macro_bias": "risk_on",
                "fetched_at": "2024-01-01T12:00:00Z",
            }
        }


class RedditSentimentItem(BaseModel):
    """Single ticker sentiment entry from Apewisdom."""

    ticker: str = Field(..., description="Ticker symbol.", example="AAPL")
    mention_count: int = Field(..., ge=0, description="Number of mentions.", example=123)
    sentiment_score: Optional[float] = Field(
        default=None,
        description="Sentiment score ranging from -1 (negative) to 1 (positive).",
        ge=-1,
        le=1,
        example=0.27,
    )

    @validator("ticker")
    def _ticker_upper(cls, v):
        return v.upper()


class RedditSentimentResponse(BaseModel):
    """Response payload for Reddit sentiment request."""

    results: List[RedditSentimentItem] = Field(
        default_factory=list,
        description="List of sentiment items (max 20).",
    )
    fetched_at: datetime = Field(
        description="Timestamp of retrieval (UTC).",
        example="2024-01-01T12:00:00Z",
    )
    source: str = Field(
        default="apewisdom.io (reddit wsb)",
        description="Data source identifier.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the fetch failed.",
        example="Apewisdom unavailable",
    )

    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {"ticker": "AAPL", "mention_count": 150, "sentiment_score": 0.3},
                    {"ticker": "TSLA", "mention_count": 120, "sentiment_score": -0.1},
                ],
                "fetched_at": "2024-01-01T12:00:00Z",
                "source": "apewisdom.io (reddit wsb)",
                "error": None,
            }
        }


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
    results = await asyncio.gather(
        _fred_latest("T10Y2Y"),
        _fred_latest("VIXCLS"),
        _fred_latest("DFF"),
        _fred_latest("BAMLH0A0HYM2"),
        _fred_latest("DTWEXBGS"),
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
        signals.credit_stress = hy_spread > 5.0
        signals.hy_spread_pct = hy_spread

    macro_score = 0
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
    return snapshot.dict()


async def get_reddit_sentiment(tickers: list[str] | None = None) -> dict:
    """
    Fetch WallStreetBets / Reddit sentiment from Apewisdom (free, no key required).
    Returns top mentioned tickers + mention count + sentiment score.
    """
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(APEWISDOM_URL, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return RedditSentimentResponse(
                        results=[],
                        fetched_at=datetime.now(timezone.utc),
                        error="Apewisdom unavailable",
                    ).dict()
                data = await resp.json()
                raw_results = data.get("results", [])
                if tickers:
                    ticker_set = {t.upper() for t in tickers}
                    raw_results = [
                        r for r in raw_results if r.get("ticker", "").upper() in ticker_set
                    ]
                items = []
                for r in raw_results[:20]:
                    try:
                        item = RedditSentimentItem(
                            ticker=r.get("ticker", ""),
                            mention_count=int(r.get("mention_count", 0)),
                            sentiment_score=r.get("sentiment_score"),
                        )
                        items.append(item)
                    except Exception:
                        # Skip malformed entries
                        continue
                response = RedditSentimentResponse(
                    results=items,
                    fetched_at=datetime.now(timezone.utc),
                )
                return response.dict()
    except Exception as e:
        logger.debug(f"Apewisdom fetch error: {e}")
        return RedditSentimentResponse(
            results=[],
            fetched_at=datetime.now(timezone.utc),
            error=str(e),
        ).dict()


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

__all__ = [
    "MacroSnapshot",
    "MacroSignals",
    "RedditSentimentResponse",
    "RedditSentimentItem",
    "get_macro_snapshot",
    "get_macro_snapshot_cached",
    "get_reddit_sentiment",
]