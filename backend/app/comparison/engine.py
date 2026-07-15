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


# Simple in‑memory cache for async benchmark fetches
_benchmark_cache: Dict[Tuple[date, date], dict] = {}
_stats_cache: Dict[Tuple, dict] = {}


class StrategyComparisonEngine:
    async def _get_benchmark_curves(self, start: date, end: date) -> dict:
        key = (start, end)
        if key not in _benchmark_cache:
            _benchmark_cache[key] = await fetch_benchmark_curves(start, end)
        return _benchmark_cache[key]

    @lru_cache(maxsize=32)
    def _get_benchmark_stats(self) -> dict:
        # get_benchmark_stats is assumed pure; cache its result
        return get_benchmark_stats()

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
        # Early exit if signals are identical – avoid duplicate backtests
        if manual_signals.equals(ml_signals):
            manual_metrics = ml_metrics = run_backtest(manual_signals, prices, initial_equity)
        else:
            manual_metrics = run_backtest(manual_signals, prices, initial_equity)
            ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        # Cache benchmark data
        benchmark_curves = await self._get_benchmark_curves(start_date, end_date)
        benchmark_stats = self._get_benchmark_stats()

        # Vectorized extraction of equity curves
        manual_eq = pd.Series(
            np.fromiter((e["equity"] for e in manual_metrics.equity_curve), dtype=float)
        )
        ml_eq = pd.Series(
            np.fromiter((e["equity"] for e in ml_metrics.equity_curve), dtype=float)
        )

        # Compute returns and perform t‑test only on overlapping period
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()
        min_len = min(len(manual_ret), len(ml_ret))

        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(
                ml_ret.iloc[:min_len].to_numpy(),
                manual_ret.iloc[:min_len].to_numpy(),
                equal_var=False,
            )
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
            is_significant=p_val < 0.05,
            winner=winner,
        )