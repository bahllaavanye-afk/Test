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

# API constants
ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_TIMEOUT = 15.0
HTTP_CLIENT_TIMEOUT = 20.0
TIMEFRAME = "1Day"
MAX_BARS_LIMIT = 1500
ALPACA_BARS_ENDPOINT_TEMPLATE = "/v2/stocks/{symbol}/bars"
ALPACA_HEADERS_KEY_ID = "APCA-API-KEY-ID"
ALPACA_HEADERS_SECRET_KEY = "APCA-API-SECRET-KEY"
ALPACA_RESPONSE_BARS_KEY = "bars"

# Normalization constants
NORMALIZATION_BASE = 100
NORMALIZATION_PRECISION = 2

# All‑Weather constants
RESAMPLE_RULE = "ME"
MIN_AW_TICKERS = 3
ALL_WEATHER_KEY = "ALL_WEATHER"

# Logging message constants
LOG_MSG_INVALID_RANGE = "Invalid benchmark date range"
LOG_MSG_FETCH_FAILED = "Alpaca bars fetch failed"
LOG_MSG_HTTP_ERROR = "HTTP error while fetching Alpaca bars"
LOG_MSG_DATA_PARSING_ERROR = "Data parsing error while processing Alpaca response"
LOG_MSG_UNEXPECTED_ERROR = "Unexpected error while fetching Alpaca bars"
LOG_MSG_ERROR_FETCH_TICKER = "Error fetching ticker data"

BENCHMARKS = {
    "SPY": {"name": "S&P 500", "color": "#2196F3"},
    "QQQ": {"name": "NASDAQ 100", "color": "#9C27B0"},
    "BRK-B": {"name": "Warren Buffett (BRK.B)", "color": "#FF9800"},
    "GLD": {"name": "Gold", "color": "#FFC107"},
}

ALL_WEATHER_WEIGHTS = {"TLT": 0.40, "IEF": 0.15, "VTI": 0.30, "GLD": 0.075, "DJP": 0.075}

# simple in‑memory cache for benchmark results keyed by (start, end)
_benchmark_cache: dict[tuple[date, date], dict[str, List[dict]]] = {}


@functools.lru_cache(maxsize=1)
def _alpaca_headers() -> dict:
    """Static Alpaca authentication headers."""
    return {
        ALPACA_HEADERS_KEY_ID: settings.alpaca_api_key,
        ALPACA_HEADERS_SECRET_KEY: settings.alpaca_secret_key,
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
            f"{ALPACA_DATA_URL}{ALPACA_BARS_ENDPOINT_TEMPLATE.format(symbol=sym)}",
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
                LOG_MSG_FETCH_FAILED,
                extra={"ticker": ticker, "status_code": resp.status_code},
            )
            return pd.Series(dtype=float)

        raw_bars = resp.json().get(ALPACA_RESPONSE_BARS_KEY, [])
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
            LOG_MSG_HTTP_ERROR,
            extra={"ticker": ticker, "error": str(exc)},
        )
        return pd.Series(dtype=float)
    except (ValueError, KeyError) as exc:
        logger.error(
            LOG_MSG_DATA_PARSING_ERROR,
            extra={"ticker": ticker, "error": str(exc)},
        )
        return pd.Series(dtype=float)
    except Exception as exc:  # pragma: no cover
        logger.exception(
            LOG_MSG_UNEXPECTED_ERROR,
            extra={"ticker": ticker},
        )
        return pd.Series(dtype=float)


async def fetch_benchmark_curves(start: date, end: date) -> dict[str, List[dict]]:
    """Returns {ticker: [{date, value}, ...]} normalized to 100 at start."""
    if start >= end:
        logger.warning(
            LOG_MSG_INVALID_RANGE,
            extra={"start": start.isoformat(), "end": end.isoformat()},
        )
        return {}

    cache_key = (start, end)
    if cached := _benchmark_cache.get(cache_key):
        # Return a shallow copy to avoid accidental mutation by callers
        return {k: v.copy() for k, v in cached.items()}

    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())

    async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
        raw_series = await asyncio.gather(
            *[_fetch_ticker_bars(client, t, start, end) for t in all_tickers],
            return_exceptions=True,
        )

    # Convert any exceptions returned by gather into empty Series and log them
    series_list: List[pd.Series] = []
    for ticker, result in zip(all_tickers, raw_series):
        if isinstance(result, Exception):
            logger.error(
                LOG_MSG_ERROR_FETCH_TICKER,
                extra={"ticker": ticker, "error": str(result)},
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
        normalized = (
            series.dropna() / series.iloc[0] * NORMALIZATION_BASE
        ).round(NORMALIZATION_PRECISION)
        result[ticker] = [
            {"date": idx.date().isoformat(), "value": float(v)} for idx, v in normalized.items()
        ]

    # All Weather: monthly rebalanced weighted portfolio
    aw_tickers = [t for t in ALL_WEATHER_WEIGHTS if t in closes_dict]
    if len(aw_tickers) >= MIN_AW_TICKERS:
        aw_frames = {t: closes_dict[t].rename(t) for t in aw_tickers}
        aw_prices = pd.concat(aw_frames.values(), axis=1).dropna()
        weights = pd.Series({t: ALL_WEATHER_WEIGHTS[t] for t in aw_tickers})
        weights = weights / weights.sum()  # renormalize if any tickers missing
        monthly_returns = aw_prices.resample(RESAMPLE_RULE).last().pct_change().dropna()
        aw_ret = (monthly_returns * weights).sum(axis=1)
        aw_equity = (1 + aw_ret).cumprod() * NORMALIZATION_BASE
        result[ALL_WEATHER_KEY] = [
            {"date": idx.date().isoformat(), "value": round(float(v), NORMALIZATION_PRECISION)} for idx, v in aw_equity.items()
        ]

    # Cache the result for future identical requests
    _benchmark_cache[cache_key] = {k: v.copy() for k, v in result.items()}
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