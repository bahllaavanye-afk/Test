"""Download benchmark equity curves via Alpaca historical bars API.
Benchmarks: SPY, QQQ, BRK-B, GLD + Ray Dalio All Weather (rebalanced monthly).
"""
from __future__ import annotations

import asyncio
import functools
from datetime import date, datetime
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple

import httpx
import pandas as pd

from app.config import settings
from app.utils.logging import logger

# Constants
ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_TIMEOUT = 15.0
HTTP_CLIENT_TIMEOUT = 20.0
TIMEFRAME = "1Day"
MAX_BARS_LIMIT = 1500
NORMALIZATION_BASE = 100
NORMALIZATION_PRECISION = 2
RESAMPLE_RULE = "ME"
MIN_AW_TICKERS = 3
ALL_WEATHER_KEY = "ALL_WEATHER"

BENCHMARKS = {
    "SPY": {"name": "S&P 500", "color": "#2196F3"},
    "QQQ": {"name": "NASDAQ 100", "color": "#9C27B0"},
    "BRK-B": {"name": "Warren Buffett (BRK.B)", "color": "#FF9800"},
    "GLD": {"name": "Gold", "color": "#FFC107"},
}

ALL_WEATHER_WEIGHTS = {"TLT": 0.40, "IEF": 0.15, "VTI": 0.30, "GLD": 0.075, "DJP": 0.075}

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
    Returns a pd.Series indexed by date, or an empty Series on failure.
    """
    sym = ticker.upper()
    start_str = datetime.combine(start, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = datetime.combine(end, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = await client.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{sym}/bars",
            params={
                "timeframe": TIMEFRAME,
                "start": start_str,
                "end": end_str,
                "limit": MAX_BARS_LIMIT,
            },
            headers=_alpaca_headers(),
            timeout=ALPACA_TIMEOUT,
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
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "Unexpected error while fetching Alpaca bars",
            extra={"ticker": ticker},
        )
        return pd.Series(dtype=float)


def _validate_date_range(start: date, end: date) -> bool:
    """Return True if the range is valid; log a warning otherwise."""
    if start >= end:
        logger.warning(
            "Invalid benchmark date range",
            extra={"start": start.isoformat(), "end": end.isoformat()},
        )
        return False
    return True


def _get_cached_result(start: date, end: date) -> Optional[dict[str, List[dict]]]:
    """Retrieve a shallow copy of a cached result if present."""
    cache_key = (start, end)
    if cached := _benchmark_cache.get(cache_key):
        return {k: v.copy() for k, v in cached.items()}
    return None


async def _fetch_all_series(
    client: httpx.AsyncClient, tickers: List[str], start: date, end: date
) -> Mapping[str, pd.Series]:
    """Fetch series for all tickers and return a mapping of ticker → Series."""
    raw_results = await asyncio.gather(
        *[_fetch_ticker_bars(client, t, start, end) for t in tickers],
        return_exceptions=True,
    )
    series_map: dict[str, pd.Series] = {}
    for ticker, result in zip(tickers, raw_results):
        if isinstance(result, Exception):
            logger.error(
                "Error fetching ticker data",
                extra={"ticker": ticker, "error": str(result)},
            )
            series_map[ticker] = pd.Series(dtype=float)
        else:
            series_map[ticker] = result
    # Filter out empty series
    return {t: s for t, s in series_map.items() if not s.empty}


def _normalize_series(series: pd.Series) -> List[dict]:
    """Normalize a price series to NORMALIZATION_BASE at its first value."""
    normalized = (series.dropna() / series.iloc[0] * NORMALIZATION_BASE).round(NORMALIZATION_PRECISION)
    return [
        {"date": idx.date().isoformat(), "value": float(v)} for idx, v in normalized.items()
    ]


def _process_all_weather(closes_dict: Mapping[str, pd.Series]) -> Optional[List[dict]]:
    """Calculate the All‑Weather portfolio equity curve if enough components exist."""
    aw_tickers = [t for t in ALL_WEATHER_WEIGHTS if t in closes_dict]
    if len(aw_tickers) < MIN_AW_TICKERS:
        return None

    aw_frames = {t: closes_dict[t].rename(t) for t in aw_tickers}
    aw_prices = pd.concat(aw_frames.values(), axis=1).dropna()
    weights = pd.Series({t: ALL_WEATHER_WEIGHTS[t] for t in aw_tickers})
    weights = weights / weights.sum()  # renormalize if any tickers missing

    monthly_returns = aw_prices.resample(RESAMPLE_RULE).last().pct_change().dropna()
    aw_ret = (monthly_returns * weights).sum(axis=1)
    aw_equity = (1 + aw_ret).cumprod() * NORMALIZATION_BASE

    return [
        {"date": idx.date().isoformat(), "value": round(float(v), NORMALIZATION_PRECISION)}
        for idx, v in aw_equity.items()
    ]


def _cache_result(start: date, end: date, result: Mapping[str, List[dict]]) -> None:
    """Store a deep copy of the result for future identical requests."""
    cache_key = (start, end)
    _benchmark_cache[cache_key] = {k: v.copy() for k, v in result.items()}


async def fetch_benchmark_curves(start: date, end: date) -> dict[str, List[dict]]:
    """Return benchmark equity curves normalized to 100 at the start date.

    The result includes individual benchmarks and the Ray Dalio All Weather portfolio.
    """
    if not _validate_date_range(start, end):
        return {}

    # Return cached data if available
    if cached := _get_cached_result(start, end):
        return cached

    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())

    async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
        closes_dict = await _fetch_all_series(client, all_tickers, start, end)

    result: dict[str, List[dict]] = {}

    # Process individual benchmarks
    for ticker in BENCHMARKS:
        if series := closes_dict.get(ticker):
            result[ticker] = _normalize_series(series)

    # Process All Weather portfolio
    if aw_curve := _process_all_weather(closes_dict):
        result[ALL_WEATHER_KEY] = aw_curve

    # Cache and return
    _cache_result(start, end, result)
    return result


# Static benchmark reference stats for display.
BENCHMARK_STATS = {
    "SPY": {"name": "S&P 500", "annual_return": 0.100, "sharpe": 0.47, "max_dd": -0.57},
    "QQQ": {"name": "NASDAQ 100", "annual_return": 0.145, "sharpe": 0.61, "max_dd": -0.83},
    "BRK-B": {"name": "Warren Buffett (BRK.B)", "annual_return": 0.199, "sharpe": 0.79, "max_dd": -0.48},
    ALL_WEATHER_KEY: {"name": "Ray Dalio All Weather", "annual_return": 0.082, "sharpe": 0.67, "max_dd": -0.20},
}


def get_benchmark_stats() -> dict:
    """Static benchmark reference stats for display."""
    return BENCHMARK_STATS