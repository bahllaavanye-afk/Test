"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger

# Simple in‑memory async cache for benchmark curves
_BENCHMARK_CACHE: Dict[Tuple[date, date], dict] = {}
_BENCHMARK_LOCK = asyncio.Lock()


async def _cached_fetch_benchmark_curves(start_date: date, end_date: date) -> dict:
    """Fetch benchmark curves with a cheap in‑process cache."""
    key = (start_date, end_date)
    async with _BENCHMARK_LOCK:
        if key not in _BENCHMARK_CACHE:
            _BENCHMARK_CACHE[key] = await fetch_benchmark_curves(start_date, end_date)
        return _BENCHMARK_CACHE[key]


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
        # Run backtests (CPU‑intensive, assumed unavoidable)
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        # Cached benchmark retrieval
        benchmark_curves = await _cached_fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Compute Sharpe improvement early; skip heavy return calculations if not significant
        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = "ml" if ml_metrics.sharpe > manual_metrics.sharpe else "manual"
        if abs(improvement) < 0.1:
            winner = "neither"

        # Default statistical values
        t_stat, p_val = 0.0, 1.0

        # Only compute returns and perform t‑test when a meaningful Sharpe gap exists
        if winner != "neither":
            # Vectorized extraction of equity series
            manual_eq_arr = np.fromiter((e["equity"] for e in manual_metrics.equity_curve), dtype=float)
            ml_eq_arr = np.fromiter((e["equity"] for e in ml_metrics.equity_curve), dtype=float)

            # Compute daily returns using NumPy for speed
            manual_ret = np.diff(manual_eq_arr) / manual_eq_arr[:-1]
            ml_ret = np.diff(ml_eq_arr) / ml_eq_arr[:-1]

            # Ensure enough observations for a reliable test
            min_len = min(manual_ret.size, ml_ret.size)
            if min_len > 10:
                # Slice to common length and use SciPy's vectorized test
                t_stat, p_val = stats.ttest_ind(ml_ret[:min_len], manual_ret[:min_len], equal_var=False)
            else:
                t_stat, p_val = 0.0, 1.0

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