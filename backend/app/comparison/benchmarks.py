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
# Unit tests for edge‑case behavior
# ----------------------------------------------------------------------
import pytest
from datetime import timedelta


@pytest.mark.asyncio
async def test_fetch_benchmark_curves_invalid_range(monkeypatch, caplog):
    """Start date equal to end date should return an empty dict and log a warning."""
    start = date(2023, 1, 1)
    end = start
    caplog.set_level("WARNING")
    result = await fetch_benchmark_curves(start, end)
    assert result == {}
    # Verify that a warning was emitted about the invalid range
    warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("Invalid benchmark date range" in w.getMessage() for w in warnings)


@pytest.mark.asyncio
async def test_fetch_benchmark_curves_all_weather_insufficient(monkeypatch):
    """ALL_WEATHER should be omitted when fewer than three constituent tickers are available."""
    start = date(2023, 1, 1)
    end = date(2023, 1, 10)

    # Mock _fetch_ticker_bars to return data only for two of the ALL_WEATHER components
    async def mock_fetch(client, ticker, s, e):
        if ticker in {"TLT", "IEF"}:
            dates = pd.date_range(s, e, freq="D", tz=timezone.utc)
            return pd.Series([100 + i for i in range(len(dates))], index=dates, name=ticker)
        return pd.Series(dtype=float)

    monkeypatch.setattr("backend.app.comparison.benchmarks._fetch_ticker_bars", mock_fetch)

    result = await fetch_benchmark_curves(start, end)
    # Ensure standard benchmarks may appear but ALL_WEATHER does not
    assert "ALL_WEATHER" not in result
    # Verify that the two mocked tickers are not in the result (they are only used for the composite)
    assert "TLT" not in result
    assert "IEF" not in result


@pytest.mark.asyncio
async def test_fetch_benchmark_curves_non_200_response(monkeypatch, caplog):
    """Tickers that receive a non‑200 response should be excluded from the output."""
    start = date(2023, 1, 1)
    end = date(2023, 1, 5)

    class MockResponse:
        def __init__(self, status_code, json_data=None):
            self.status_code = status_code
            self._json = json_data or {}

        def json(self):
            return self._json

    class MockClient:
        async def get(self, url, params=None, headers=None, timeout=None):
            # Simulate a 404 for SPY, success for others
            if "SPY" in url:
                return MockResponse(404)
            # Return a minimal valid payload for other tickers
            return MockResponse(
                200,
                {"bars": [{"t": "2023-01-02T00:00:00Z", "c": 101.0}]},
            )

    # Patch the AsyncClient used inside fetch_benchmark_curves
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: MockClient())

    caplog.set_level("WARNING")
    result = await fetch_benchmark_curves(start, end)
    # SPY should be missing because of the 404 response
    assert "SPY" not in result
    # At least one other benchmark (e.g., QQQ) should be present
    assert any(ticker in result for ticker in {"QQQ", "BRK-B", "GLD"})


# The tests are defined in the same module to keep the repository self‑contained.
# They can be discovered by pytest when the module is imported as part of the test suite.