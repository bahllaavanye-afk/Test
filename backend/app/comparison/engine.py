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
        manual_ret, ml_ret = self._compute_return_series(manual_metrics, ml_metrics)
        t_stat, p_val = self._perform_t_test(manual_ret, ml_ret)
        improvement, winner = self._determine_winner(manual_metrics, ml_metrics)
        self._log_comparison(strategy_name, manual_metrics, ml_metrics, p_val)

        return self._build_result(
            strategy_name,
            symbol,
            interval,
            start_date,
            end_date,
            manual_metrics,
            ml_metrics,
            benchmark_curves,
            benchmark_stats,
            improvement,
            t_stat,
            p_val,
            winner,
        )

    def _run_backtests(
        self,
        manual_signals: pd.Series,
        ml_signals: pd.Series,
        prices: pd.Series,
        initial_equity: float,
    ) -> tuple[BacktestMetrics, BacktestMetrics]:
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)
        return manual_metrics, ml_metrics

    async def _fetch_benchmarks(
        self, start_date: date, end_date: date
    ) -> tuple[dict, dict]:
        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()
        return benchmark_curves, benchmark_stats

    def _compute_return_series(
        self, manual_metrics: BacktestMetrics, ml_metrics: BacktestMetrics
    ) -> tuple[pd.Series, pd.Series]:
        manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()
        return manual_ret, ml_ret

    def _perform_t_test(
        self, manual_ret: pd.Series, ml_ret: pd.Series
    ) -> tuple[float, float]:
        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
        else:
            t_stat, p_val = 0.0, 1.0
        return float(t_stat), float(p_val)

    def _determine_winner(
        self, manual_metrics: BacktestMetrics, ml_metrics: BacktestMetrics
    ) -> tuple[float, str]:
        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = "ml" if ml_metrics.sharpe > manual_metrics.sharpe else "manual"
        if abs(improvement) < 0.1:
            winner = "neither"
        return improvement, winner

    def _log_comparison(
        self,
        strategy_name: str,
        manual_metrics: BacktestMetrics,
        ml_metrics: BacktestMetrics,
        p_val: float,
    ) -> None:
        logger.info(
            "Comparison complete",
            strategy=strategy_name,
            manual_sharpe=manual_metrics.sharpe,
            ml_sharpe=ml_metrics.sharpe,
            p_value=round(p_val, 4),
        )

    def _build_result(
        self,
        strategy_name: str,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
        manual_metrics: BacktestMetrics,
        ml_metrics: BacktestMetrics,
        benchmark_curves: dict,
        benchmark_stats: dict,
        improvement: float,
        t_stat: float,
        p_val: float,
        winner: str,
    ) -> ComparisonResult:
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
            t_statistic=round(t_stat, 4),
            p_value=round(p_val, 6),
            is_significant=(p_val < 0.05),
            winner=winner,
        )