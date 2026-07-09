"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger

# Simple in‑memory caches for benchmark data to avoid repeated I/O
_BENCHMARK_CURVES_CACHE: Dict[Tuple[date, date], dict] = {}
_BENCHMARK_STATS_CACHE: Dict[bool, dict] = {}


def _cached_fetch_benchmark_curves(start: date, end: date) -> dict:
    """Fetch benchmark curves with a lightweight in‑process cache."""
    key = (start, end)
    if key not in _BENCHMARK_CURVES_CACHE:
        # fetch_benchmark_curves is async; we run it in the event loop here.
        loop = asyncio.get_event_loop()
        _BENCHMARK_CURVES_CACHE[key] = loop.run_until_complete(fetch_benchmark_curves(start, end))
    return _BENCHMARK_CURVES_CACHE[key]


def _cached_get_benchmark_stats() -> dict:
    """Cache static benchmark statistics."""
    if not _BENCHMARK_STATS_CACHE:
        _BENCHMARK_STATS_CACHE[True] = get_benchmark_stats()
    return _BENCHMARK_STATS_CACHE[True]


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
        # Run backtests (potentially expensive; kept as‑is)
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        # Cached benchmark data
        benchmark_curves = await fetch_benchmark_curves(start_date, end_date) if (start_date, end_date) not in _BENCHMARK_CURVES_CACHE else _BENCHMARK_CURVES_CACHE[(start_date, end_date)]
        _BENCHMARK_CURVES_CACHE[(start_date, end_date)] = benchmark_curves
        benchmark_stats = _cached_get_benchmark_stats()

        # Vectorized extraction of equity curves and returns
        manual_eq = np.fromiter((e["equity"] for e in manual_metrics.equity_curve), dtype=float)
        ml_eq = np.fromiter((e["equity"] for e in ml_metrics.equity_curve), dtype=float)

        # Compute simple returns using NumPy (avoids pandas overhead)
        manual_ret = np.diff(manual_eq) / manual_eq[:-1]
        ml_ret = np.diff(ml_eq) / ml_eq[:-1]

        # Ensure enough observations for statistical test
        min_len = min(manual_ret.size, ml_ret.size)
        if min_len > 10:
            # Use the first min_len observations for a fair comparison
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