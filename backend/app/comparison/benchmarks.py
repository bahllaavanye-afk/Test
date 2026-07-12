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
from pydantic import BaseModel, Field, validator

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


class BenchmarkPoint(BaseModel):
    """A single data point representing a benchmark value on a specific date."""

    date: date = Field(
        ...,
        description="Date of the benchmark observation in ISO format (YYYY‑MM‑DD).",
        example="2023-01-01",
    )
    value: float = Field(
        ...,
        ge=0,
        description="Benchmark value normalized to 100 at the start date.",
        example=102.5,
    )

    @validator("date")
    def date_not_in_future(cls, v: date) -> date:
        """Ensure the date is not in the future relative to UTC now."""
        if v > datetime.now(timezone.utc).date():
            raise ValueError("Benchmark date cannot be in the future.")
        return v


class BenchmarkStat(BaseModel):
    """Static reference statistics for a benchmark."""

    name: str = Field(
        ...,
        description="Human‑readable name of the benchmark.",
        example="S&P 500",
    )
    annual_return: float = Field(
        ...,
        description="Annualized return expressed as a decimal (e.g., 0.10 for 10%).",
        example=0.10,
    )
    sharpe: float = Field(
        ...,
        description="Sharpe ratio of the benchmark.",
        example=0.47,
    )
    max_dd: float = Field(
        ...,
        description="Maximum drawdown expressed as a decimal (negative values).",
        example=-0.57,
    )

    @validator("annual_return")
    def return_reasonable_range(cls, v: float) -> float:
        if not -1.0 <= v <= 2.0:
            raise ValueError("Annual return out of expected range.")
        return v

    @validator("sharpe")
    def sharpe_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Sharpe ratio should be non‑negative.")
        return v

    @validator("max_dd")
    def max_dd_negative(cls, v: float) -> float:
        if v > 0:
            raise ValueError("Maximum drawdown should be negative or zero.")
        return v


# Type aliases for clearer signatures
BenchmarkCurvesResponse = Dict[str, List[BenchmarkPoint]]
BenchmarkStatsResponse = Dict[str, BenchmarkStat]


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


async def fetch_benchmark_curves(start: date, end: date) -> BenchmarkCurvesResponse:
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

    result: dict[str, List[BenchmarkPoint]] = {}

    # Process individual benchmarks
    for ticker in BENCHMARKS:
        series = closes_dict.get(ticker)
        if series is None or series.empty:
            continue
        normalized = (series.dropna() / series.iloc[0] * 100).round(2)
        result[ticker] = [
            BenchmarkPoint(date=idx.date(), value=float(v))
            for idx, v in normalized.items()
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
            BenchmarkPoint(date=idx.date(), value=round(float(v), 2))
            for idx, v in aw_equity.items()
        ]

    # Cache the result for future identical requests
    _benchmark_cache[cache_key] = {k: v.copy() for k, v in result.items()}
    return result


def get_benchmark_stats() -> BenchmarkStatsResponse:
    """Static benchmark reference stats for display."""
    return {
        "SPY": BenchmarkStat(
            name="S&P 500",
            annual_return=0.100,
            sharpe=0.47,
            max_dd=-0.57,
        ),
        "QQQ": BenchmarkStat(
            name="NASDAQ 100",
            annual_return=0.145,
            sharpe=0.61,
            max_dd=-0.83,
        ),
        "BRK-B": BenchmarkStat(
            name="Warren Buffett (BRK.B)",
            annual_return=0.199,
            sharpe=0.79,
            max_dd=-0.48,
        ),
        "ALL_WEATHER": BenchmarkStat(
            name="Ray Dalio All Weather",
            annual_return=0.082,
            sharpe=0.67,
            max_dd=-0.20,
        ),
    }