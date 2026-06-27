"""
Download benchmark equity curves via Alpaca historical bars API.
Benchmarks: SPY, QQQ, BRK-B, GLD + Ray Dalio All Weather (rebalanced monthly).
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any, Dict, List, Mapping

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


def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


class BenchmarkPoint(BaseModel):
    """Single point of a benchmark equity curve."""

    date: date = Field(..., description="Date of the observation", example="2024-01-02")
    value: float = Field(
        ...,
        description="Normalized equity value (base = 100)",
        example=102.34,
        ge=0,
    )

    @validator("value")
    def no_nan(cls, v: float) -> float:
        if pd.isna(v):
            raise ValueError("value must not be NaN")
        return v


class BenchmarkCurveResponse(BaseModel):
    """Mapping of ticker symbols to their equity curve data."""

    __root__: Mapping[str, List[BenchmarkPoint]] = Field(
        ...,
        description="Dictionary where each key is a ticker and each value is a list of equity points",
    )

    class Config:
        schema_extra = {
            "example": {
                "SPY": [
                    {"date": "2024-01-02", "value": 100.0},
                    {"date": "2024-01-03", "value": 101.2},
                ],
                "ALL_WEATHER": [
                    {"date": "2024-01-02", "value": 100.0},
                    {"date": "2024-02-01", "value": 101.5},
                ],
            }
        }


class BenchmarkStatItem(BaseModel):
    """Static reference statistics for a benchmark."""

    name: str = Field(..., description="Human‑readable name of the benchmark", example="S&P 500")
    annual_return: float = Field(
        ...,
        description="Annualized total return (as a decimal)",
        example=0.10,
        ge=-1,
        le=5,
    )
    sharpe: float = Field(
        ...,
        description="Sharpe ratio",
        example=0.47,
        ge=-10,
        le=10,
    )
    max_dd: float = Field(
        ...,
        description="Maximum drawdown (as a decimal, negative)",
        example=-0.57,
        le=0,
    )


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
        resp.raise_for_status()
    except httpx.HTTPError as http_err:
        logger.error(
            "HTTP error while fetching Alpaca bars",
            ticker=ticker,
            error=str(http_err),
            url=resp.url if "resp" in locals() else None,
        )
        return pd.Series(dtype=float)
    except Exception as exc:
        logger.error("Unexpected error during Alpaca request", ticker=ticker, error=str(exc))
        return pd.Series(dtype=float)

    try:
        raw_bars = resp.json().get("bars", [])
    except ValueError as json_err:
        logger.error("Failed to parse JSON response from Alpaca", ticker=ticker, error=str(json_err))
        return pd.Series(dtype=float)

    if not raw_bars:
        logger.warning("No bar data returned from Alpaca", ticker=ticker)
        return pd.Series(dtype=float)

    try:
        dates = pd.to_datetime([b["t"] for b in raw_bars], utc=True).normalize()
        closes = [float(b["c"]) for b in raw_bars]
    except (KeyError, TypeError, ValueError) as parse_err:
        logger.error(
            "Error parsing bar data",
            ticker=ticker,
            error=str(parse_err),
        )
        return pd.Series(dtype=float)

    series = pd.Series(closes, index=dates, name=ticker)
    series = series[~series.index.duplicated(keep="last")]
    return series


async def fetch_benchmark_curves(start: date, end: date) -> BenchmarkCurveResponse:
    """Returns normalized equity curves for each benchmark ticker.

    The curves are expressed as a list of ``BenchmarkPoint`` objects and are
    normalized to a value of 100 at the start date.
    """
    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())

    async with httpx.AsyncClient(timeout=20.0) as client:
        raw_results = await asyncio.gather(
            *[_fetch_ticker_bars(client, t, start, end) for t in all_tickers],
            return_exceptions=True,
        )

    series_list: List[pd.Series] = []
    for ticker, result in zip(all_tickers, raw_results):
        if isinstance(result, Exception):
            logger.error(
                "Failed to fetch ticker bars",
                ticker=ticker,
                error=str(result),
            )
            continue
        series_list.append(result)

    closes_dict: Dict[str, pd.Series] = {
        ticker: series
        for ticker, series in zip(all_tickers, series_list)
        if not series.empty
    }

    result: Dict[str, List[BenchmarkPoint]] = {}

    for ticker in BENCHMARKS:
        if ticker not in closes_dict:
            continue
        series = closes_dict[ticker].dropna()
        if series.empty:
            continue
        normalized = (series / series.iloc[0] * 100).round(2)
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
        weights = weights / weights.sum()
        monthly_returns = aw_prices.resample("ME").last().pct_change().dropna()
        aw_ret = (monthly_returns * weights).sum(axis=1)
        aw_equity = (1 + aw_ret).cumprod() * 100
        result["ALL_WEATHER"] = [
            BenchmarkPoint(date=idx.date(), value=round(float(v), 2))
            for idx, v in aw_equity.items()
        ]

    return BenchmarkCurveResponse(__root__=result)


def get_benchmark_stats() -> Dict[str, BenchmarkStatItem]:
    """Static benchmark reference stats for display."""
    raw = {
        "SPY": {"name": "S&P 500", "annual_return": 0.100, "sharpe": 0.47, "max_dd": -0.57},
        "QQQ": {"name": "NASDAQ 100", "annual_return": 0.145, "sharpe": 0.61, "max_dd": -0.83},
        "BRK-B": {"name": "Warren Buffett (BRK.B)", "annual_return": 0.199, "sharpe": 0.79, "max_dd": -0.48},
        "ALL_WEATHER": {"name": "Ray Dalio All Weather", "annual_return": 0.082, "sharpe": 0.67, "max_dd": -0.20},
    }
    return {k: BenchmarkStatItem(**v) for k, v in raw.items()}