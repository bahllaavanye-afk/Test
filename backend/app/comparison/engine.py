"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date

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
        manual_metrics, ml_metrics = self._run_backtests(
            manual_signals, ml_signals, prices, initial_equity
        )

        benchmark_curves, benchmark_stats = await self._fetch_benchmarks(
            start_date, end_date
        )

        manual_ret, ml_ret = self._extract_return_series(manual_metrics, ml_metrics)

        t_stat, p_val = self._perform_ttest(manual_ret, ml_ret)

        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = self._determine_winner(ml_metrics.sharpe, manual_metrics.sharpe, improvement)

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

    def _run_backtests(
        self,
        manual_signals: pd.Series,
        ml_signals: pd.Series,
        prices: pd.Series,
        initial_equity: float,
    ) -> tuple[BacktestMetrics, BacktestMetrics]:
        """Execute backtests for manual and ML‑enhanced signals."""
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)
        return manual_metrics, ml_metrics

    async def _fetch_benchmarks(
        self, start_date: date, end_date: date
    ) -> tuple[dict, dict]:
        """Retrieve benchmark curves and statistics."""
        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()
        return benchmark_curves, benchmark_stats

    def _extract_return_series(
        self, manual_metrics: BacktestMetrics, ml_metrics: BacktestMetrics
    ) -> tuple[pd.Series, pd.Series]:
        """Convert equity curves to daily return series."""
        manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()
        return manual_ret, ml_ret

    def _perform_ttest(
        self, manual_ret: pd.Series, ml_ret: pd.Series
    ) -> tuple[float, float]:
        """Run a two‑sample t‑test if sufficient data points are available."""
        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
        else:
            t_stat, p_val = 0.0, 1.0
        return t_stat, p_val

    def _determine_winner(
        self, ml_sharpe: float, manual_sharpe: float, improvement: float
    ) -> str:
        """Select the winning strategy based on Sharpe ratio improvement."""
        winner = "ml" if ml_sharpe > manual_sharpe else "manual"
        if abs(improvement) < 0.1:
            winner = "neither"
        return winner