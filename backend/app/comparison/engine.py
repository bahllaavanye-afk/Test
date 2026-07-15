"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import date
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Extract daily return series for t-test
        manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()

        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
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


# ==========================
# Unit Tests for Edge Cases
# ==========================
class TestStrategyComparisonEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = StrategyComparisonEngine()
        self.start_date = date(2023, 1, 1)
        self.end_date = date(2023, 1, 31)

        # Minimal mock for benchmark functions
        self.benchmark_curves_mock = AsyncMock(return_value={})
        self.benchmark_stats_mock = MagicMock(return_value={})

    async def test_short_return_series_defaults(self):
        """When return series length <= 10, t-statistic should default to 0 and p-value to 1."""
        manual_signals = pd.Series([1, 0, 1])
        ml_signals = pd.Series([0, 1, 0])
        prices = pd.Series([100, 101, 102])

        # Mock BacktestMetrics with short equity curves (3 points)
        mock_metrics = MagicMock()
        mock_metrics.equity_curve = [{"equity": 100_000}, {"equity": 100_500}, {"equity": 101_000}]
        mock_metrics.sharpe = 0.5

        with patch("app.backtest.engine.run_backtest", return_value=mock_metrics), \
             patch("app.comparison.benchmarks.fetch_benchmark_curves", self.benchmark_curves_mock), \
             patch("app.comparison.benchmarks.get_benchmark_stats", self.benchmark_stats_mock):
            result = await self.engine.run_comparison(
                manual_signals, ml_signals, prices,
                strategy_name="test_strategy",
                symbol="TEST",
                interval="1d",
                start_date=self.start_date,
                end_date=self.end_date,
            )
        self.assertEqual(result.t_statistic, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.is_significant)

    async def test_improvement_boundary(self):
        """Improvement exactly at 0.1 should be considered significant and not default to 'neither'."""
        manual_signals = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        ml_signals = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        prices = pd.Series(np.arange(100, 112))

        # Create equity curves long enough (>10 returns) and Sharpe diff exactly 0.1
        manual_metrics = MagicMock()
        manual_metrics.equity_curve = [{"equity": 100_000 + i * 1000} for i in range(13)]
        manual_metrics.sharpe = 0.8

        ml_metrics = MagicMock()
        ml_metrics.equity_curve = [{"equity": 100_000 + i * 1100} for i in range(13)]
        ml_metrics.sharpe = 0.9  # diff = 0.1

        with patch("app.backtest.engine.run_backtest", side_effect=[manual_metrics, ml_metrics]), \
             patch("app.comparison.benchmarks.fetch_benchmark_curves", self.benchmark_curves_mock), \
             patch("app.comparison.benchmarks.get_benchmark_stats", self.benchmark_stats_mock):
            result = await self.engine.run_comparison(
                manual_signals, ml_signals, prices,
                strategy_name="boundary_test",
                symbol="BND",
                interval="1d",
                start_date=self.start_date,
                end_date=self.end_date,
            )
        # Since improvement == 0.1 (not < 0.1), winner should be 'ml'
        self.assertEqual(result.winner, "ml")
        self.assertAlmostEqual(result.ml_improvement_sharpe, 0.1, places=4)

    async def test_equal_sharpe_results_in_neither_winner(self):
        """When both strategies have identical Sharpe, winner should be 'neither' due to zero improvement."""
        manual_signals = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        ml_signals = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        prices = pd.Series(np.linspace(100, 200, 12))

        # Equity curves with identical returns
        shared_curve = [{"equity": 100_000 + i * 1000} for i in range(13)]

        manual_metrics = MagicMock()
        manual_metrics.equity_curve = shared_curve
        manual_metrics.sharpe = 1.0

        ml_metrics = MagicMock()
        ml_metrics.equity_curve = shared_curve
        ml_metrics.sharpe = 1.0

        with patch("app.backtest.engine.run_backtest", side_effect=[manual_metrics, ml_metrics]), \
             patch("app.comparison.benchmarks.fetch_benchmark_curves", self.benchmark_curves_mock), \
             patch("app.comparison.benchmarks.get_benchmark_stats", self.benchmark_stats_mock):
            result = await self.engine.run_comparison(
                manual_signals, ml_signals, prices,
                strategy_name="equal_sharpe",
                symbol="EQ",
                interval="1d",
                start_date=self.start_date,
                end_date=self.end_date,
            )
        self.assertEqual(result.winner, "neither")
        self.assertAlmostEqual(result.ml_improvement_sharpe, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()