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
    # Simple in‑memory cache for benchmark curves to avoid repeated async fetches
    _benchmark_cache: Dict[Tuple[date, date], dict] = {}

    async def _get_benchmark_curves(self, start_date: date, end_date: date) -> dict:
        cache_key = (start_date, end_date)
        if cache_key not in self._benchmark_cache:
            self._benchmark_cache[cache_key] = await fetch_benchmark_curves(start_date, end_date)
        return self._benchmark_cache[cache_key]

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
        # Run backtests concurrently; use thread executor because run_backtest is CPU bound
        if manual_signals.equals(ml_signals):
            # Identical signals – reuse result
            manual_metrics = await asyncio.to_thread(run_backtest, manual_signals, prices, initial_equity)
            ml_metrics = manual_metrics
        else:
            manual_task = asyncio.to_thread(run_backtest, manual_signals, prices, initial_equity)
            ml_task = asyncio.to_thread(run_backtest, ml_signals, prices, initial_equity)
            manual_metrics, ml_metrics = await asyncio.gather(manual_task, ml_task)

        # Cache benchmark curves to reduce network/IO latency
        benchmark_curves = await self._get_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Convert equity curves to numpy arrays for fast return calculation
        manual_eq = np.array([e["equity"] for e in manual_metrics.equity_curve], dtype=float)
        ml_eq = np.array([e["equity"] for e in ml_metrics.equity_curve], dtype=float)

        # Compute daily returns using vectorized NumPy operations
        manual_ret = np.diff(manual_eq) / manual_eq[:-1]
        ml_ret = np.diff(ml_eq) / ml_eq[:-1]

        # Align lengths for t‑test
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