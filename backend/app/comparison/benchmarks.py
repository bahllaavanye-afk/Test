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

BENCHMARKS = {
    "SPY": {"name": "S&P 500", "color": "#2196F3"},
    "QQQ": {"name": "NASDAQ 100", "color": "#9C27B0"},
    "BRK-B": {"name": "Warren Buffett (BRK.B)", "color": "#FF9800"},
    "GLD": {"name": "Gold", "color": "#FFC107"},
}

ALL_WEATHER_WEIGHTS = {
    "TLT": 0.40,
    "IEF": 0.15,
    "VTI": 0.30,
    "GLD": 0.075,
    "DJP": 0.075,
}

ALPACA_DATA_URL = "https://data.alpaca.markets"

# simple in‑memory cache for benchmark results keyed by (start, end)
_benchmark_cache: dict[tuple[date, date], dict[str, List[dict]]] = {}


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

    Returns a ``pd.Series`` indexed by date.  On any failure an empty
    ``Series`` is returned and the error is logged with structured context.
    """
    sym = ticker.upper()
    start_str = datetime.combine(start, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = datetime.combine(end, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = await client.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{sym}/bars",
            params={
                "timeframe": "1Day",
                "start": start_str,
                "end": end_str,
                "limit": 1500,
            },
            headers=_alpaca_headers(),
            timeout=15.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Alpaca bars HTTP error",
            ticker=ticker,
            status_code=exc.response.status_code,
            url=str(exc.request.url),
            error=str(exc),
        )
        return pd.Series(dtype=float)
    except httpx.RequestError as exc:
        logger.error(
            "Alpaca request failed",
            ticker=ticker,
            url=str(exc.request.url) if exc.request else None,
            error=str(exc),
        )
        return pd.Series(dtype=float)

    try:
        raw_bars = resp.json().get("bars", [])
    except ValueError as exc:
        logger.error(
            "Failed to decode Alpaca JSON response",
            ticker=ticker,
            error=str(exc),
        )
        return pd.Series(dtype=float)

    if not raw_bars:
        logger.warning("Alpaca returned empty bar list", ticker=ticker)
        return pd.Series(dtype=float)

    try:
        dates = pd.to_datetime([b["t"] for b in raw_bars], utc=True).normalize()
        closes = [float(b["c"]) for b in raw_bars]
    except (KeyError, TypeError, ValueError) as exc:
        logger.error(
            "Unexpected bar format from Alpaca",
            ticker=ticker,
            error=str(exc),
        )
        return pd.Series(dtype=float)

    series = pd.Series(closes, index=dates, name=ticker)
    # De‑duplicate any same‑day entries (take last)
    series = series[~series.index.duplicated(keep="last")]
    return series


async def fetch_benchmark_curves(start: date, end: date) -> dict[str, List[dict]]:
    """Returns {ticker: [{date, value}, ...]} normalized to 100 at start."""
    if start >= end:
        logger.warning("Invalid date range for benchmark fetch", start=start.isoformat(), end=end.isoformat())
        return {}

    cache_key = (start, end)
    if cached := _benchmark_cache.get(cache_key):
        # Return a shallow copy to avoid accidental mutation by callers
        return {k: v.copy() for k, v in cached.items()}

    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())

    async with httpx.AsyncClient(timeout=20.0) as client:
        # Gather with return_exceptions to isolate failures per ticker
        raw_series = await asyncio.gather(
            *[_fetch_ticker_bars(client, t, start, end) for t in all_tickers],
            return_exceptions=True,
        )

    series_list: List[pd.Series] = []
    for ticker, result in zip(all_tickers, raw_series):
        if isinstance(result, Exception):
            logger.error(
                "Failed to fetch bars for ticker",
                ticker=ticker,
                error=str(result),
            )
            series_list.append(pd.Series(dtype=float))
        else:
            series_list.append(result)

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
        normalized = (series.dropna() / series.iloc[0] * 100).round(2)
        result[ticker] = [
            {"date": idx.date().isoformat(), "value": float(v)} for idx, v in normalized.items()
        ]

    # All Weather: monthly rebalanced weighted portfolio
    aw_tickers = [t for t in ALL_WEATHER_WEIGHTS if t in closes_dict]
    if len(aw_tickers) >= 3:
        aw_frames = {t: closes_dict[t].rename(t) for t in aw_tickers}
        aw_prices = pd.concat(aw_frames.values(), axis=1).dropna()
        weights = pd.Series({t: ALL_WEATHER_WEIGHTS[t] for t in aw_tickers})
        weights = weights / weights.sum()  # renormalize if any tickers missing
        monthly_returns = aw_prices.resample("ME").last().pct_change().dropna()
        aw_ret = (monthly_returns * weights).sum(axis=1)
        aw_equity = (1 + aw_ret).cumprod() * 100
        result["ALL_WEATHER"] = [
            {"date": idx.date().isoformat(), "value": round(float(v), 2)} for idx, v in aw_equity.items()
        ]

    # Cache the result for future identical requests
    _benchmark_cache[cache_key] = {k: v.copy() for k, v in result.items()}
    return result


def get_benchmark_stats() -> dict:
    """Static benchmark reference stats for display."""
    return {
        "SPY": {"name": "S&P 500", "annual_return": 0.100, "sharpe": 0.47, "max_dd": -0.57},
        "QQQ": {"name": "NASDAQ 100", "annual_return": 0.145, "sharpe": 0.61, "max_dd": -0.83},
        "BRK-B": {"name": "Warren Buffett (BRK.B)", "annual_return": 0.199, "sharpe": 0.79, "max_dd": -0.48},
        "ALL_WEATHER": {"name": "Ray Dalio All Weather", "annual_return": 0.082, "sharpe": 0.67, "max_dd": -0.20},
    }