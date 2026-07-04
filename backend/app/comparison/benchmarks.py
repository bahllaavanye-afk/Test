"""Download benchmark equity curves via Alpaca historical bars API.
Benchmarks: SPY, QQQ, BRK-B, GLD + Ray Dalio All Weather (rebalanced monthly).
"""
from __future__ import annotations

import asyncio
import functools
from datetime import date, datetime
from typing import Dict, List, Tuple

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
_benchmark_cache: dict[Tuple[date, date], dict[str, List[dict]]] = {}


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
    start_str = datetime.combine(start, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = datetime.combine(end, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = await client.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{sym}/bars",
            params={"timeframe": "1Day", "start": start_str, "end": end_str, "limit": 1500},
            headers=_alpaca_headers(),
            timeout=15.0,
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


def _validate_date_range(start: date, end: date) -> bool:
    """Return True if the date range is non‑empty and ordered correctly."""
    return start < end


def _get_cached_result(cache_key: Tuple[date, date]) -> dict[str, List[dict]] | None:
    """Return a shallow copy of cached data if present."""
    if cached := _benchmark_cache.get(cache_key):
        return {k: v.copy() for k, v in cached.items()}
    return None


def _cache_result(cache_key: Tuple[date, date], result: dict[str, List[dict]]) -> None:
    """Store a shallow copy of result in the in‑memory cache."""
    _benchmark_cache[cache_key] = {k: v.copy() for k, v in result.items()}


def _normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a price series to 100 at its first observation."""
    return (series.dropna() / series.iloc[0] * 100).round(2)


def _build_result_entry(series: pd.Series) -> List[dict]:
    """Convert a normalized series into the API‑compatible list of dicts."""
    return [
        {"date": idx.date().isoformat(), "value": float(v)} for idx, v in series.items()
    ]


def _compute_all_weather(closes: Dict[str, pd.Series]) -> List[dict]:
    """
    Compute the All Weather portfolio equity curve (monthly rebalanced).
    Returns a list of {date, value} dicts.
    """
    aw_tickers = [t for t in ALL_WEATHER_WEIGHTS if t in closes]
    if len(aw_tickers) < 3:
        return []

    # Align price frames and drop any rows with missing data
    aw_frames = {t: closes[t].rename(t) for t in aw_tickers}
    aw_prices = pd.concat(aw_frames.values(), axis=1).dropna()

    # Adjust weights for any missing tickers
    weights = pd.Series({t: ALL_WEATHER_WEIGHTS[t] for t in aw_tickers})
    weights = weights / weights.sum()

    # Monthly returns based on end‑of‑month prices
    monthly_returns = aw_prices.resample("ME").last().pct_change().dropna()
    portfolio_ret = (monthly_returns * weights).sum(axis=1)

    # Cumulative equity curve starting at 100
    equity = (1 + portfolio_ret).cumprod() * 100
    return [
        {"date": idx.date().isoformat(), "value": round(float(v), 2)} for idx, v in equity.items()
    ]


async def fetch_benchmark_curves(start: date, end: date) -> dict[str, List[dict]]:
    """Returns {ticker: [{date, value}, ...]} normalized to 100 at start."""
    if not _validate_date_range(start, end):
        return {}

    cache_key = (start, end)
    if cached_result := _get_cached_result(cache_key):
        return cached_result

    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())

    async with httpx.AsyncClient(timeout=20.0) as client:
        series_list = await asyncio.gather(
            *[_fetch_ticker_bars(client, t, start, end) for t in all_tickers]
        )

    # Map ticker -> series, discarding empty results
    closes_dict: dict[str, pd.Series] = {
        ticker: series
        for ticker, series in zip(all_tickers, series_list)
        if not series.empty
    }

    result: dict[str, List[dict]] = {}

    # Process individual benchmark tickers
    for ticker in BENCHMARKS:
        if series := closes_dict.get(ticker):
            normalized = _normalize_series(series)
            result[ticker] = _build_result_entry(normalized)

    # Process All Weather portfolio
    aw_curve = _compute_all_weather(closes_dict)
    if aw_curve:
        result["ALL_WEATHER"] = aw_curve

    _cache_result(cache_key, result)
    return result


def get_benchmark_stats() -> dict:
    """Static benchmark reference stats for display."""
    return {
        "SPY": {"name": "S&P 500", "annual_return": 0.100, "sharpe": 0.47, "max_dd": -0.57},
        "QQQ": {"name": "NASDAQ 100", "annual_return": 0.145, "sharpe": 0.61, "max_dd": -0.83},
        "BRK-B": {"name": "Warren Buffett (BRK.B)", "annual_return": 0.199, "sharpe": 0.79, "max_dd": -0.48},
        "ALL_WEATHER": {"name": "Ray Dalio All Weather", "annual_return": 0.082, "sharpe": 0.67, "max_dd": -0.20},
    }