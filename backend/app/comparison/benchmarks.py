"""Download benchmark equity curves via Alpaca historical bars API.
Benchmarks: SPY, QQQ, BRK-B, GLD + Ray Dalio All Weather (rebalanced monthly).
"""
from __future__ import annotations

import asyncio
import functools
from datetime import date, datetime, timezone
from typing import Dict, List

import httpx
import pandas as pd

from app.config import settings
from app.utils.logging import logger

# Constants
ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_TIMEFRAME = "1Day"
ALPACA_LIMIT = 1500
ALPACA_TIMEOUT = 15.0
ALPACA_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

HTTP_CLIENT_TIMEOUT = 20.0

NORMALIZATION_BASE = 100.0
NORMALIZATION_DECIMALS = 2

ALL_WEATHER_MIN_TICKERS = 3
ALL_WEATHER_RESAMPLE_RULE = "ME"
ALL_WEATHER_ROUND_DECIMALS = 2

BENCHMARKS = {
    "SPY": {"name": "S&P 500", "color": "#2196F3"},
    "QQQ": {"name": "NASDAQ 100", "color": "#9C27B0"},
    "BRK-B": {"name": "Warren Buffett (BRK.B)", "color": "#FF9800"},
    "GLD": {"name": "Gold", "color": "#FFC107"},
}

ALL_WEATHER_WEIGHTS = {"TLT": 0.40, "IEF": 0.15, "VTI": 0.30, "GLD": 0.075, "DJP": 0.075}

# simple in‑memory cache for benchmark results keyed by (start, end)
_benchmark_cache: dict[tuple[date, date], dict[str, List[dict]]] = {}

BENCHMARK_STATS = {
    "SPY": {"name": "S&P 500", "annual_return": 0.100, "sharpe": 0.47, "max_dd": -0.57},
    "QQQ": {"name": "NASDAQ 100", "annual_return": 0.145, "sharpe": 0.61, "max_dd": -0.83},
    "BRK-B": {"name": "Warren Buffett (BRK.B)", "annual_return": 0.199, "sharpe": 0.79, "max_dd": -0.48},
    "ALL_WEATHER": {"name": "Ray Dalio All Weather", "annual_return": 0.082, "sharpe": 0.67, "max_dd": -0.20},
}


@functools.lru_cache(maxsize=1)
def _alpaca_headers() -> dict:
    """Static Alpaca authentication headers."""
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


async def _fetch_ticker_bars(
    client: httpx.AsyncClient, ticker: str, start: date, end: date
) -> pd.Series:
    """
    Fetch daily close prices for a single ticker from Alpaca.
    Returns a pd.Series indexed by date, or empty Series on failure.
    """
    sym = ticker.upper()
    start_str = datetime.combine(start, datetime.min.time()).strftime(ALPACA_DATE_FORMAT)
    end_str = datetime.combine(end, datetime.min.time()).strftime(ALPACA_DATE_FORMAT)

    try:
        resp = await client.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{sym}/bars",
            params={
                "timeframe": ALPACA_TIMEFRAME,
                "start": start_str,
                "end": end_str,
                "limit": ALPACA_LIMIT,
            },
            headers=_alpaca_headers(),
            timeout=ALPACA_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("Alpaca bars fetch failed", ticker=ticker, status=resp.status_code)
            return pd.Series(dtype=float)

        raw_bars = resp.json().get("bars", [])
        if not raw_bars:
            return pd.Series(dtype=float)

        dates = pd.to_datetime([b["t"] for b in raw_bars], utc=True).normalize()
        closes = [float(b["c"]) for b in raw_bars]
        series = pd.Series(closes, index=dates, name=ticker)
        # De‑duplicate any same‑day entries (take last)
        series = series[~series.index.duplicated(keep="last")]
        return series

    except Exception as exc:  # pragma: no cover
        logger.warning("Alpaca bars exception", ticker=ticker, error=str(exc))
        return pd.Series(dtype=float)


async def fetch_benchmark_curves(start: date, end: date) -> dict[str, List[dict]]:
    """Returns {ticker: [{date, value}, ...]} normalized to 100 at start."""
    if start >= end:
        return {}

    cache_key = (start, end)
    if cached := _benchmark_cache.get(cache_key):
        # Return a shallow copy to avoid accidental mutation by callers
        return {k: v.copy() for k, v in cached.items()}

    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())

    async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
        series_list = await asyncio.gather(
            *[_fetch_ticker_bars(client, t, start, end) for t in all_tickers]
        )

    closes_dict: dict[str, pd.Series] = {
        ticker: series
        for ticker, series in zip(all_tickers, series_list)
        if not series.empty
    }

    result: dict[str, List[dict]] = {}

    # Process individual benchmarks
    for ticker in BENCHMARKS:
        series = closes_dict.get(ticker)
        if series is None or series.empty:
            continue
        normalized = (series.dropna() / series.iloc[0] * NORMALIZATION_BASE).round(NORMALIZATION_DECIMALS)
        result[ticker] = [
            {"date": idx.date().isoformat(), "value": float(v)} for idx, v in normalized.items()
        ]

    # All Weather: monthly rebalanced weighted portfolio
    aw_tickers = [t for t in ALL_WEATHER_WEIGHTS if t in closes_dict]
    if len(aw_tickers) >= ALL_WEATHER_MIN_TICKERS:
        aw_frames = {t: closes_dict[t].rename(t) for t in aw_tickers}
        aw_prices = pd.concat(aw_frames.values(), axis=1).dropna()
        weights = pd.Series({t: ALL_WEATHER_WEIGHTS[t] for t in aw_tickers})
        weights = weights / weights.sum()  # renormalize if any tickers missing
        monthly_returns = aw_prices.resample(ALL_WEATHER_RESAMPLE_RULE).last().pct_change().dropna()
        aw_ret = (monthly_returns * weights).sum(axis=1)
        aw_equity = (1 + aw_ret).cumprod() * NORMALIZATION_BASE
        result["ALL_WEATHER"] = [
            {"date": idx.date().isoformat(), "value": round(float(v), ALL_WEATHER_ROUND_DECIMALS)} for idx, v in aw_equity.items()
        ]

    # Cache the result for future identical requests
    _benchmark_cache[cache_key] = {k: v.copy() for k, v in result.items()}
    return result


def get_benchmark_stats() -> dict:
    """Static benchmark reference stats for display."""
    return BENCHMARK_STATS