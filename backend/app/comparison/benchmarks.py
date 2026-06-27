"""
Download benchmark equity curves via Alpaca historical bars API.
Benchmarks: SPY, QQQ, BRK-B, GLD + Ray Dalio All Weather (rebalanced monthly).
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any, Dict, List

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


def _alpaca_headers() -> dict:
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


async def fetch_benchmark_curves(start: date, end: date) -> Dict[str, List[Dict[str, Any]]]:
    """Returns {ticker: [{date, value}, ...]} normalized to 100 at start."""
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

    result: Dict[str, List[Dict[str, Any]]] = {}

    for ticker in BENCHMARKS:
        if ticker not in closes_dict:
            continue
        series = closes_dict[ticker].dropna()
        if series.empty:
            continue
        normalized = (series / series.iloc[0] * 100).round(2)
        result[ticker] = [
            {"date": str(idx.date()), "value": float(v)} for idx, v in normalized.items()
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
            {"date": str(idx.date()), "value": round(float(v), 2)} for idx, v in aw_equity.items()
        ]

    return result


def get_benchmark_stats() -> dict:
    """Static benchmark reference stats for display."""
    return {
        "SPY": {"name": "S&P 500", "annual_return": 0.100, "sharpe": 0.47, "max_dd": -0.57},
        "QQQ": {"name": "NASDAQ 100", "annual_return": 0.145, "sharpe": 0.61, "max_dd": -0.83},
        "BRK-B": {"name": "Warren Buffett (BRK.B)", "annual_return": 0.199, "sharpe": 0.79, "max_dd": -0.48},
        "ALL_WEATHER": {"name": "Ray Dalio All Weather", "annual_return": 0.082, "sharpe": 0.67, "max_dd": -0.20},
    }