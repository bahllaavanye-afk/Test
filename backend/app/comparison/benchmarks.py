"""
Download benchmark equity curves via Alpaca historical bars API.
Benchmarks: SPY, QQQ, BRK-B, GLD + Ray Dalio All Weather (rebalanced monthly).
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Tuple

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

# Simple in‑memory cache for fetched series to avoid duplicate network calls
_SERIES_CACHE: Dict[Tuple[str, date, date], pd.Series] = {}
_CACHE_LOCK = asyncio.Lock()


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
            url=getattr(resp, "url", None),
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
        logger.error("Error parsing bar data", ticker=ticker, error=str(parse_err))
        return pd.Series(dtype=float)

    series = pd.Series(closes, index=dates, name=ticker)
    series = series[~series.index.duplicated(keep="last")]
    return series


def _validate_date_range(start: date, end: date) -> None:
    """Validate that start and end are proper dates and that start <= end."""
    if not isinstance(start, date):
        raise ValueError(f"start must be a datetime.date instance, got {type(start).__name__}")
    if not isinstance(end, date):
        raise ValueError(f"end must be a datetime.date instance, got {type(end).__name__}")
    if start > end:
        raise ValueError(f"start date {start} must not be after end date {end}")


async def _cached_fetch_ticker_series(
    client: httpx.AsyncClient, ticker: str, start: date, end: date
) -> pd.Series:
    """
    Wrapper that caches the result of _fetch_ticker_bars to avoid duplicate requests.
    """
    cache_key = (ticker.upper(), start, end)
    async with _CACHE_LOCK:
        if cache_key in _SERIES_CACHE:
            return _SERIES_CACHE[cache_key]

    series = await _fetch_ticker_bars(client, ticker, start, end)

    async with _CACHE_LOCK:
        _SERIES_CACHE[cache_key] = series
    return series


async def _fetch_all_ticker_series(
    client: httpx.AsyncClient, tickers: List[str], start: date, end: date
) -> Dict[str, pd.Series]:
    """
    Concurrently fetch bar series for all tickers.
    Returns a mapping of ticker -> non‑empty pd.Series.
    """
    tasks = [
        _cached_fetch_ticker_series(client, t, start, end) for t in tickers
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    series_dict: Dict[str, pd.Series] = {}
    for ticker, result in zip(tickers, raw_results):
        if isinstance(result, Exception):
            logger.error(
                "Failed to fetch ticker bars",
                ticker=ticker,
                error=str(result),
            )
            continue
        if result.empty:
            continue
        series_dict[ticker] = result
    return series_dict


def _normalize_series(series: pd.Series) -> List[BenchmarkPoint]:
    """
    Normalize a price series to start at 100 and convert to BenchmarkPoint list.
    """
    series = series.dropna()
    if series.empty:
        return []
    normalized = (series / series.iloc[0] * 100).round(2)
    return [
        BenchmarkPoint(date=idx.date(), value=float(v))
        for idx, v in normalized.items()
    ]


def _build_all_weather_curve(closes_dict: Dict[str, pd.Series]) -> List[BenchmarkPoint]:
    """
    Construct the All Weather portfolio curve (monthly rebalanced) from available series.
    Returns an empty list if insufficient data.
    """
    available_tickers = [t for t in ALL_WEATHER_WEIGHTS if t in closes_dict]
    if len(available_tickers) < 3:
        return []

    # Align price series
    price_frames = {t: closes_dict[t].rename(t) for t in available_tickers}
    prices = pd.concat(price_frames.values(), axis=1).dropna()

    # Normalise weights to sum to 1
    weights = pd.Series({t: ALL_WEATHER_WEIGHTS[t] for t in available_tickers})
    weights = weights / weights.sum()

    # Monthly rebalancing using month‑end prices
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    portfolio_ret = (monthly_returns * weights).sum(axis=1)

    equity_curve = (1 + portfolio_ret).cumprod() * 100
    return [
        BenchmarkPoint(date=idx.date(), value=round(float(v), 2))
        for idx, v in equity_curve.items()
    ]


async def fetch_benchmark_curves(start: date, end: date) -> BenchmarkCurveResponse:
    """Returns normalized equity curves for each benchmark ticker.

    The curves are expressed as a list of ``BenchmarkPoint`` objects and are
    normalized to a value of 100 at the start date.
    """
    _validate_date_range(start, end)

    async with httpx.AsyncClient() as client:
        series_map = await _fetch_all_ticker_series(
            client, list(BENCHMARKS.keys()), start, end
        )

    curves: Dict[str, List[BenchmarkPoint]] = {
        ticker: _normalize_series(series)
        for ticker, series in series_map.items()
    }

    # Build All Weather curve from the same series map (may contain extra tickers)
    all_weather_curve = _build_all_weather_curve(series_map)
    if all_weather_curve:
        curves["ALL_WEATHER"] = all_weather_curve

    return BenchmarkCurveResponse(__root__=curves)