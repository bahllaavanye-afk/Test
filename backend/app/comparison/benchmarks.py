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

ALL_WEATHER_WEIGHTS = {
    "TLT": 0.40,
    "IEF": 0.15,
    "VTI": 0.30,
    "GLD": 0.075,
    "DJP": 0.075,
}

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
    """Return True if the date range is valid, otherwise log a warning."""
    if start >= end:
        logger.warning(
            "Invalid benchmark date range",
            extra={"start": start.isoformat(), "end": end.isoformat()},
        )
        return False
    return True


async def _fetch_all_tickers(
    client: httpx.AsyncClient, tickers: List[str], start: date, end: date
) -> dict[str, pd.Series]:
    """
    Retrieve close price series for each ticker.
    Errors are logged and result in an empty Series.
    """
    raw_results = await asyncio.gather(
        *[_fetch_ticker_bars(client, t, start, end) for t in tickers],
        return_exceptions=True,
    )

    series_dict: dict[str, pd.Series] = {}
    for ticker, result in zip(tickers, raw_results):
        if isinstance(result, Exception):
            logger.error(
                "Error fetching ticker data",
                extra={"ticker": ticker, "error": str(result)},
            )
            series_dict[ticker] = pd.Series(dtype=float)
        else:
            series_dict[ticker] = result
    # Remove empty series
    return {t: s for t, s in series_dict.items() if not s.empty}


def _process_individual_benchmarks(
    closes: dict[str, pd.Series]
) -> dict[str, List[dict]]:
    """
    Normalize each benchmark series to 100 at the start date and format for output.
    """
    result: dict[str, List[dict]] = {}
    for ticker in BENCHMARKS:
        series = closes.get(ticker)
        if series is None or series.empty:
            continue
        normalized = (
            series.dropna() / series.iloc[0] * NORMALIZATION_BASE
        ).round(NORMALIZATION_PRECISION)
        result[ticker] = [
            {"date": idx.date().isoformat(), "value": float(v)}
            for idx, v in normalized.items()
        ]
    return result


def _process_all_weather(
    closes: dict[str, pd.Series]
) -> dict[str, List[dict]]:
    """
    Build the Ray Dalio All Weather portfolio, rebalanced monthly.
    Returns a dict with a single entry keyed by ``ALL_WEATHER_KEY``.
    """
    aw_tickers = [t for t in ALL_WEATHER_WEIGHTS if t in closes]
    if len(aw_tickers) < MIN_AW_TICKERS:
        return {}

    aw_frames = {t: closes[t].rename(t) for t in aw_tickers}
    aw_prices = pd.concat(aw_frames.values(), axis=1).dropna()
    weights = pd.Series({t: ALL_WEATHER_WEIGHTS[t] for t in aw_tickers})
    weights = weights / weights.sum()  # renormalize if any tickers missing

    monthly_returns = aw_prices.resample(RESAMPLE_RULE).last().pct_change().dropna()
    aw_ret = (monthly_returns * weights).sum(axis=1)
    aw_equity = (1 + aw_ret).cumprod() * NORMALIZATION_BASE

    return {
        ALL_WEATHER_KEY: [
            {
                "date": idx.date().isoformat(),
                "value": round(float(v), NORMALIZATION_PRECISION),
            }
            for idx, v in aw_equity.items()
        ]
    }


def _cache_result(
    cache_key: tuple[date, date], data: dict[str, List[dict]]
) -> None:
    """Store a shallow copy of the result in the in‑memory cache."""
    _benchmark_cache[cache_key] = {k: v.copy() for k, v in data.items()}


async def fetch_benchmark_curves(start: date, end: date) -> dict[str, List[dict]]:
    """Returns {ticker: [{date, value}, ...]} normalized to 100 at start."""
    if not _validate_date_range(start, end):
        return {}

    cache_key = (start, end)
    if cached := _benchmark_cache.get(cache_key):
        # Return a shallow copy to avoid accidental mutation by callers
        return {k: v.copy() for k, v in cached.items()}

    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())

    async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
        closes = await _fetch_all_tickers(client, all_tickers, start, end)

    result = _process_individual_benchmarks(closes)
    result.update(_process_all_weather(closes))

    _cache_result(cache_key, result)
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