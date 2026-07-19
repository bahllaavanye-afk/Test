"""Download benchmark equity curves via Alpaca historical bars API.
Benchmarks: SPY, QQQ, BRK-B, GLD + Ray Dalio All Weather (rebalanced monthly).
"""
from __future__ import annotations

import asyncio
import functools
from datetime import date, datetime
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

ALL_WEATHER_WEIGHTS = {"TLT": 0.40, "IEF": 0.15, "VTI": 0.30, "GLD": 0.075, "DJP": 0.075}

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
    Returns a pd.Series indexed by date, or an empty Series on failure.
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
            logger.warning(
                "Alpaca bars fetch failed",
                extra={"ticker": ticker, "status_code": resp.status_code},
            )
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

    except httpx.HTTPError as exc:
        logger.error(
            "HTTP error while fetching Alpaca bars",
            extra={"ticker": ticker, "error": str(exc)},
        )
        return pd.Series(dtype=float)
    except (ValueError, KeyError) as exc:
        logger.error(
            "Data parsing error while processing Alpaca response",
            extra={"ticker": ticker, "error": str(exc)},
        )
        return pd.Series(dtype=float)
    except Exception:  # pragma: no cover
        logger.exception(
            "Unexpected error while fetching Alpaca bars",
            extra={"ticker": ticker},
        )
        return pd.Series(dtype=float)


def _validate_date_range(start: date, end: date) -> bool:
    """Return True if the date range is valid; log a warning otherwise."""
    if start >= end:
        logger.warning(
            "Invalid benchmark date range",
            extra={"start": start.isoformat(), "end": end.isoformat()},
        )
        return False
    return True


def _get_cached_result(key: tuple[date, date]) -> dict[str, List[dict]] | None:
    """Retrieve a shallow copy of a cached result, if present."""
    cached = _benchmark_cache.get(key)
    if cached:
        return {k: v.copy() for k, v in cached.items()}
    return None


async def _fetch_all_series(
    client: httpx.AsyncClient, tickers: List[str], start: date, end: date
) -> dict[str, pd.Series]:
    """Fetch series for all tickers and return a dict of non‑empty Series."""
    raw_series = await asyncio.gather(
        *[_fetch_ticker_bars(client, t, start, end) for t in tickers],
        return_exceptions=True,
    )

    series_dict: dict[str, pd.Series] = {}
    for ticker, result in zip(tickers, raw_series):
        if isinstance(result, Exception):
            logger.error(
                "Error fetching ticker data",
                extra={"ticker": ticker, "error": str(result)},
            )
            continue
        if not result.empty:
            series_dict[ticker] = result
    return series_dict


def _process_individual_benchmarks(closes_dict: dict[str, pd.Series]) -> dict[str, List[dict]]:
    """Normalize each benchmark series to 100 at the start date."""
    result: dict[str, List[dict]] = {}
    for ticker in BENCHMARKS:
        series = closes_dict.get(ticker)
        if series is None or series.empty:
            continue
        normalized = (series.dropna() / series.iloc[0] * 100).round(2)
        result[ticker] = [
            {"date": idx.date().isoformat(), "value": float(v)} for idx, v in normalized.items()
        ]
    return result


def _process_all_weather(closes_dict: dict[str, pd.Series]) -> List[dict]:
    """Compute the monthly rebalanced All‑Weather portfolio equity curve."""
    aw_tickers = [t for t in ALL_WEATHER_WEIGHTS if t in closes_dict]
    if len(aw_tickers) < 3:
        return []

    aw_frames = {t: closes_dict[t].rename(t) for t in aw_tickers}
    aw_prices = pd.concat(aw_frames.values(), axis=1).dropna()
    weights = pd.Series({t: ALL_WEATHER_WEIGHTS[t] for t in aw_tickers})
    weights = weights / weights.sum()  # renormalize if any tickers missing

    monthly_returns = aw_prices.resample("ME").last().pct_change().dropna()
    aw_ret = (monthly_returns * weights).sum(axis=1)
    aw_equity = (1 + aw_ret).cumprod() * 100

    return [
        {"date": idx.date().isoformat(), "value": round(float(v), 2)} for idx, v in aw_equity.items()
    ]


def _cache_result(key: tuple[date, date], result: dict[str, List[dict]]) -> None:
    """Store a shallow copy of the result in the in‑memory cache."""
    _benchmark_cache[key] = {k: v.copy() for k, v in result.items()}


async def fetch_benchmark_curves(start: date, end: date) -> dict[str, List[dict]]:
    """Returns {ticker: [{date, value}, ...]} normalized to 100 at start."""
    if not _validate_date_range(start, end):
        return {}

    cache_key = (start, end)
    cached = _get_cached_result(cache_key)
    if cached:
        return cached

    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())

    async with httpx.AsyncClient(timeout=20.0) as client:
        closes_dict = await _fetch_all_series(client, all_tickers, start, end)

    result = _process_individual_benchmarks(closes_dict)

    aw_curve = _process_all_weather(closes_dict)
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