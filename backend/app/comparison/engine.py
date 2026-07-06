"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from app.backtest.engine import BacktestMetrics, run_backtest
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


# Simple in‑memory cache for async benchmark fetches
_benchmark_cache: Dict[Tuple[date, date], dict] = {}


async def _cached_fetch_benchmark_curves(start: date, end: date) -> dict:
    """Fetch benchmark curves with a per‑process cache.

    The underlying ``fetch_benchmark_curves`` may involve network I/O; caching
    avoids repeated calls for the same date range during a single process
    lifetime.
    """
    key = (start, end)
    if key not in _benchmark_cache:
        _benchmark_cache[key] = await fetch_benchmark_curves(start, end)
    return _benchmark_cache[key]


@dataclass
class ComparisonResult:
    strategy_name: str
    symbol: str
    interval: str
    start_date: date
    end_date: date
    manual: BackbacktestMetrics | None = None
    ml_enhanced: BacktestMetrics | None = None
    benchmark_curves: dict = field(default_factory=dict)
    benchmark_stats: dict = field(default_factory=dict)
    ml_improvement_sharpe: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    winner: str = "neither"


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
        """Run a side‑by‑side comparison of a manual and an ML‑enhanced strategy.

        The function:
        * Executes back‑tests for both signal sets.
        * Retrieves benchmark data (cached per process).
        * Computes daily returns using NumPy for speed.
        * Performs a two‑sample t‑test when sufficient data points exist.
        * Summarises the results in a ``ComparisonResult`` instance.
        """
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await _cached_fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Convert equity curves to NumPy arrays for fast vectorised operations
        manual_eq_arr = np.fromiter(
            (e["equity"] for e in manual_metrics.equity_curve), dtype=float, count=len(manual_metrics.equity_curve)
        )
        ml_eq_arr = np.fromiter(
            (e["equity"] for e in ml_metrics.equity_curve), dtype=float, count=len(ml_metrics.equity_curve)
        )

        # Daily returns: (P_t - P_{t-1}) / P_{t-1}
        manual_ret = np.diff(manual_eq_arr) / manual_eq_arr[:-1]
        ml_ret = np.diff(ml_eq_arr) / ml_eq_arr[:-1]

        # Ensure both series have the same length for the t‑test
        min_len = min(manual_ret.size, ml_ret.size)

        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(ml_ret[:min_len], manual_ret[:min_len], equal_var=False)
        else:
            t_stat, p_val = 0.0, 1.0

        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = "ml" if ml_metrics.sharpe > manual_metrics.sharpe else "manual"
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
            is_significant=(p_val < 0.05),
            winner=winner,
        )