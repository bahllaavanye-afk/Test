"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any

import pandas as pd
from scipy import stats

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


@dataclass
class ComparisonResult:
    strategy_name: str
    symbol: str
    interval: str
    start_date: date
    end_date: date
    manual: BacktestMetrics | None = None
    ml_enhanced: BacktestMetrics | None = None
    benchmark_curves: dict = field(default_factory=dict)
    benchmark_stats: dict = field(default_factory=dict)
    ml_improvement_sharpe: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    winner: str = "neither"


# Simple async‑compatible cache for benchmark curves
def _async_lru_cache(maxsize: int = 32):
    def decorator(func):
        cache = lru_cache(maxsize=maxsize)(func)

        async def wrapper(*args, **kwargs):
            return cache(*args, **kwargs)

        return wrapper

    return decorator


@_async_lru_cache(maxsize=32)
async def _cached_fetch_benchmark_curves(start: date, end: date) -> dict:
    return await fetch_benchmark_curves(start, end)


class StrategyComparisonEngine:
    async def run_comparison(
        self,
        manual_signals: pd.Series,
        ml_signals: pd.Series,
        prices: pd.Series,
        strategy_name: str,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
        initial_equity: float = 100_000,
    ) -> ComparisonResult:
        # Input validation
        if not isinstance(manual_signals, pd.Series):
            raise ValueError("manual_signals must be a pandas Series.")
        if not isinstance(ml_signals, pd.Series):
            raise ValueError("ml_signals must be a pandas Series.")
        if not isinstance(prices, pd.Series):
            raise ValueError("prices must be a pandas Series.")
        if manual_signals.empty:
            raise ValueError("manual_signals series cannot be empty.")
        if ml_signals.empty:
            raise ValueError("ml_signals series cannot be empty.")
        if prices.empty:
            raise ValueError("prices series cannot be empty.")
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string.")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("interval must be a non-empty string.")
        if not isinstance(start_date, date):
            raise ValueError("start_date must be a datetime.date instance.")
        if not isinstance(end_date, date):
            raise ValueError("end_date must be a datetime.date instance.")
        if start_date > end_date:
            raise ValueError("start_date cannot be later than end_date.")
        if not isinstance(initial_equity, (int, float)):
            raise ValueError("initial_equity must be a numeric type.")
        if initial_equity <= 0:
            raise ValueError("initial_equity must be a positive number.")

        # Align series on common index
        common_index = manual_signals.index.intersection(ml_signals.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError("manual_signals, ml_signals, and prices must share at least one common index.")
        manual_signals = manual_signals.loc[common_index]
        ml_signals = ml_signals.loc[common_index]
        prices = prices.loc[common_index]

        # Run backtests
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        # Fetch benchmark data (cached)
        benchmark_curves = await _cached_fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Vectorized extraction of equity curves
        def _equity_series(metrics: BacktestMetrics) -> pd.Series:
            # Assume equity_curve is list‑like of dicts with an "equity" key
            if isinstance(metrics.equity_curve, pd.DataFrame):
                return metrics.equity_curve["equity"]
            return pd.Series([e["equity"] for e in metrics.equity_curve])

        manual_eq = _equity_series(manual_metrics)
        ml_eq = _equity_series(ml_metrics)

        # Compute returns using NumPy for speed
        manual_ret = manual_eq.pct_change().dropna().to_numpy()
        ml_ret = ml_eq.pct_change().dropna().to_numpy()

        # Early exit if insufficient data for statistical test
        if len(manual_ret) < 10 or len(ml_ret) < 10:
            t_stat, p_val = 0.0, 1.0
        else:
            # Align lengths
            min_len = min(len(manual_ret), len(ml_ret))
            t_stat, p_val = stats.ttest_ind(ml_ret[:min_len], manual_ret[:min_len], equal_var=False)

        # Sharpe improvement and winner determination
        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        if ml_metrics.sharpe > manual_metrics.sharpe:
            winner = "ml"
        elif manual_metrics.sharpe > ml_metrics.sharpe:
            winner = "manual"
        else:
            winner = "neither"
        if abs(improvement) < 0.1:
            winner = "neither"

        logger.info(
            "Comparison complete",
            strategy=strategy_name,
            manual_sharpe=manual_metrics.sharpe,
            ml_sharpe=ml_metrics.sharpe,
            p_value=round(p_val, 4),
        )

        return ComparisonResult(
            strategy_name=strategy_name,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            manual=manual_metrics,
            ml_enhanced=ml_metrics,
            benchmark_curves=benchmark_curves,
            benchmark_stats=benchmark_stats,
            ml_improvement_sharpe=round(improvement, 4),
            t_statistic=round(float(t_stat), 4),
            p_value=round(float(p_val), 6),
            is_significant=p_val < 0.05,
            winner=winner,
        )