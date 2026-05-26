"""
Download benchmark equity curves via yfinance.
Benchmarks: SPY, QQQ, BRK-B, GLD + Ray Dalio All Weather (rebalanced monthly).
"""
from __future__ import annotations
import asyncio
from datetime import date
from functools import lru_cache

import pandas as pd
import yfinance as yf

from app.utils.logging import logger

BENCHMARKS = {
    "SPY": {"name": "S&P 500", "color": "#2196F3"},
    "QQQ": {"name": "NASDAQ 100", "color": "#9C27B0"},
    "BRK-B": {"name": "Warren Buffett (BRK.B)", "color": "#FF9800"},
    "GLD": {"name": "Gold", "color": "#FFC107"},
}

ALL_WEATHER_WEIGHTS = {"TLT": 0.40, "IEF": 0.15, "VTI": 0.30, "GLD": 0.075, "DJP": 0.075}


def _download_sync(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    raw = yf.download(tickers, start=str(start), end=str(end), auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]] if "Close" in raw.columns else raw
    return closes.dropna(how="all")


async def fetch_benchmark_curves(start: date, end: date) -> dict[str, list[dict]]:
    """Returns {ticker: [{date, value}, ...]} normalized to 100 at start."""
    all_tickers = list(BENCHMARKS.keys()) + list(ALL_WEATHER_WEIGHTS.keys())
    loop = asyncio.get_event_loop()
    closes = await loop.run_in_executor(None, _download_sync, all_tickers, start, end)

    result: dict[str, list[dict]] = {}

    for ticker in BENCHMARKS:
        if ticker not in closes.columns:
            continue
        series = closes[ticker].dropna()
        if series.empty:
            continue
        normalized = (series / series.iloc[0] * 100).round(2)
        result[ticker] = [{"date": str(d.date()), "value": float(v)} for d, v in normalized.items()]

    # All Weather: monthly rebalanced weighted portfolio
    aw_tickers = [t for t in ALL_WEATHER_WEIGHTS if t in closes.columns]
    if len(aw_tickers) >= 3:
        aw_prices = closes[aw_tickers].dropna()
        weights = pd.Series({t: ALL_WEATHER_WEIGHTS[t] for t in aw_tickers})
        weights = weights / weights.sum()  # renormalize if any tickers missing
        monthly_returns = aw_prices.resample("ME").last().pct_change().dropna()
        aw_ret = (monthly_returns * weights).sum(axis=1)
        aw_equity = (1 + aw_ret).cumprod() * 100
        result["ALL_WEATHER"] = [{"date": str(d.date()), "value": round(float(v), 2)} for d, v in aw_equity.items()]

    return result


def get_benchmark_stats() -> dict:
    """Static benchmark reference stats for display."""
    return {
        "SPY":         {"name": "S&P 500",               "annual_return": 0.100, "sharpe": 0.47, "max_dd": -0.57},
        "QQQ":         {"name": "NASDAQ 100",             "annual_return": 0.145, "sharpe": 0.61, "max_dd": -0.83},
        "BRK-B":       {"name": "Warren Buffett (BRK.B)", "annual_return": 0.199, "sharpe": 0.79, "max_dd": -0.48},
        "ALL_WEATHER":  {"name": "Ray Dalio All Weather",  "annual_return": 0.082, "sharpe": 0.67, "max_dd": -0.20},
    }
