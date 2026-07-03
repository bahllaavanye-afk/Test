"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from app.backtest.engine import BacktestMetrics, run_backtest
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
        """Run a full comparison between manual and ML‑enhanced strategies.

        The method:
        1. Executes backtests for both signal sets.
        2. Retrieves benchmark data.
        3. Computes daily returns and performs a t‑test.
        4. Calculates Sharpe improvement and determines the winner.
        5. Returns a populated ``ComparisonResult`` instance.
        """
        manual_metrics = self._run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = self._run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await self._fetch_benchmarks(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        manual_ret = self._daily_returns(manual_metrics)
        ml_ret = self._daily_returns(ml_metrics)

        t_stat, p_val = self._perform_t_test(ml_ret, manual_ret)

        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = self._determine_winner(improvement, ml_metrics.sharpe, manual_metrics.sharpe)

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

    def _run_backtest(
        self,
        signals: pd.Series,
        prices: pd.Series,
        initial_equity: float,
    ) -> BacktestMetrics:
        """Execute a backtest and return its metrics."""
        return run_backtest(signals, prices, initial_equity)

    async def _fetch_benchmarks(self, start_date: date, end_date: date) -> dict:
        """Fetch benchmark curves for the given period."""
        return await fetch_benchmark_curves(start_date, end_date)

    def _daily_returns(self, metrics: BacktestMetrics) -> pd.Series:
        """Extract equity curve from backtest metrics and compute daily returns."""
        equity_series = pd.Series([point["equity"] for point in metrics.equity_curve])
        return equity_series.pct_change().dropna()

    def _perform_t_test(
        self,
        ml_ret: pd.Series,
        manual_ret: pd.Series,
    ) -> Tuple[float, float]:
        """Conduct an independent two‑sample t‑test on the return series.

        Returns a tuple of (t_statistic, p_value). If the overlapping sample
        length is insufficient (< 10), a neutral result (0.0, 1.0) is returned.
        """
        min_len = min(len(ml_ret), len(manual_ret))
        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
            return float(t_stat), float(p_val)
        return 0.0, 1.0

    def _determine_winner(
        self,
        improvement: float,
        ml_sharpe: float,
        manual_sharpe: float,
    ) -> str:
        """Decide which strategy is the winner based on Sharpe improvement.

        If the absolute Sharpe difference is less than 0.1, the result is
        considered a tie ("neither").
        """
        if abs(improvement) < 0.1:
            return "neither"
        return "ml" if ml_sharpe > manual_sharpe else "manual"