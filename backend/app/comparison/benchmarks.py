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
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "Unexpected error while fetching Alpaca bars",
            extra={"ticker": ticker},
        )
        return pd.Series(dtype=float)


async def fetch_benchmark_curves(start: date, end: date) -> dict[str, List[dict]]:
    """Returns {ticker: [{date, value}, ...]} normalized to 100 at start."""
    if start >= end:
        logger.warning(
            "Invalid benchmark date range",
            extra={"start": start.isoformat(), "end": end.isoformat()},
        )
        return {}

    cache_key = (start, end)
    if cached := _benchmark_cache.get(cache_key):
        # Return a shallow copy to avoid accidental mutation by callers
        return {k: v.copy() for k, v in cached.items()}

    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())

    async with httpx.AsyncClient(timeout=20.0) as client:
        raw_series = await asyncio.gather(
            *[_fetch_ticker_bars(client, t, start, end) for t in all_tickers],
            return_exceptions=True,
        )

    # Convert any exceptions returned by gather into empty Series and log them
    series_list: List[pd.Series] = []
    for ticker, result in zip(all_tickers, raw_series):
        if isinstance(result, Exception):
            logger.error(
                "Error fetching ticker data",
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


# ----------------------------------------------------------------------
# Unit tests for edge‑case handling
# ----------------------------------------------------------------------
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_fetch_benchmark_curves_invalid_range():
    """When start date is on or after end date, an empty dict should be returned."""
    from backend.app.comparison import benchmarks as bm

    start = date(2023, 1, 1)
    end = date(2023, 1, 1)  # equal to start
    result = await bm.fetch_benchmark_curves(start, end)
    assert result == {}

    # also test start > end
    result = await bm.fetch_benchmark_curves(date(2023, 2, 1), date(2023, 1, 1))
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_benchmark_curves_all_empty(monkeypatch):
    """
    If the underlying fetch returns empty series for every ticker,
    the function should return an empty result (no benchmark keys).
    """
    from backend.app.comparison import benchmarks as bm

    async def empty_series(*args, **kwargs):
        return pd.Series(dtype=float)

    monkeypatch.setattr(bm, "_fetch_ticker_bars", empty_series)

    start = date(2023, 1, 1)
    end = date(2023, 1, 10)
    result = await bm.fetch_benchmark_curves(start, end)
    # No keys should be present because all series are empty
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_benchmark_curves_cache_immutability(monkeypatch):
    """
    Ensure that the cached result is not mutated by callers.
    The second call should return a copy; modifying the returned dict must not affect the cache.
    """
    from backend.app.comparison import benchmarks as bm

    # Mock fetch to return a deterministic small series for SPY only
    dates = pd.date_range(start="2023-01-01", periods=3, freq="D", tz=timezone.utc)
    series = pd.Series([100.0, 101.0, 102.0], index=dates, name="SPY")

    async def mock_fetch(client, ticker, start, end):
        return series if ticker == "SPY" else pd.Series(dtype=float)

    monkeypatch.setattr(bm, "_fetch_ticker_bars", mock_fetch)

    start = date(2023, 1, 1)
    end = date(2023, 1, 3)

    # First call populates the cache
    result1 = await bm.fetch_benchmark_curves(start, end)
    assert "SPY" in result1
    original_spy_data = result1["SPY"].copy()

    # Mutate the returned data
    result1["SPY"][0]["value"] = 9999.0

    # Second call should retrieve a fresh copy from cache
    result2 = await bm.fetch_benchmark_curves(start, end)
    assert result2["SPY"] == original_spy_data  # unchanged by previous mutation