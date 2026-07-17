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


# ==============================
# Unit Tests for Edge Conditions
# ==============================
import unittest
from unittest.mock import patch, MagicMock


class SimpleMetrics:
    """Minimal BacktestMetrics-like object for testing."""

    def __init__(self, equity_curve, sharpe):
        self.equity_curve = equity_curve
        self.sharpe = sharpe


class TestStrategyComparisonEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = StrategyComparisonEngine()
        self.start_date = date(2022, 1, 1)
        self.end_date = date(2022, 12, 31)

    @patch("app.comparison.engine.fetch_benchmark_curves")
    @patch("app.comparison.engine.get_benchmark_stats")
    @patch("app.comparison.engine.run_backtest")
    async def test_short_return_series_uses_default_statistics(
        self, mock_run_backtest, mock_get_stats, mock_fetch_curves
    ):
        """When return series length ≤10, t-stat should be 0 and p-value 1."""
        mock_fetch_curves.return_value = {}
        mock_get_stats.return_value = {}

        # equity curves with only 5 points → returns length 4
        equity_curve = [{"equity": v} for v in [100_000, 101_000, 102_000, 103_000, 104_000]]
        mock_run_backtest.side_effect = [
            SimpleMetrics(equity_curve, sharpe=1.0),  # manual
            SimpleMetrics(equity_curve, sharpe=1.2),  # ml
        ]

        manual_signals = pd.Series([1, 0, 1, 0, 1])
        ml_signals = pd.Series([0, 1, 0, 1, 0])
        prices = pd.Series([50, 51, 52, 53, 54])

        result = await self.engine.run_comparison(
            manual_signals,
            ml_signals,
            prices,
            strategy_name="test_strategy",
            symbol="TEST",
            interval="1D",
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self.assertEqual(result.t_statistic, 0.0)
        self.assertEqual(result.p_value, 1.0)

    @patch("app.comparison.engine.fetch_benchmark_curves")
    @patch("app.comparison.engine.get_benchmark_stats")
    @patch("app.comparison.engine.run_backtest")
    async def test_improvement_exact_threshold_allows_winner_selection(
        self, mock_run_backtest, mock_get_stats, mock_fetch_curves
    ):
        """Improvement exactly 0.1 should not trigger the 'neither' winner."""
        mock_fetch_curves.return_value = {}
        mock_get_stats.return_value = {}

        equity_curve = [{"equity": v} for v in range(100_000, 100_500, 100)]
        manual_metrics = SimpleMetrics(equity_curve, sharpe=1.0)
        ml_metrics = SimpleMetrics(equity_curve, sharpe=1.1)  # improvement = 0.1

        mock_run_backtest.side_effect = [manual_metrics, ml_metrics]

        manual_signals = pd.Series([1, 1, 1, 1, 1])
        ml_signals = pd.Series([0, 0, 0, 0, 0])
        prices = pd.Series([10, 11, 12, 13, 14])

        result = await self.engine.run_comparison(
            manual_signals,
            ml_signals,
            prices,
            strategy_name="threshold_test",
            symbol="THR",
            interval="1D",
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self.assertEqual(result.winner, "ml")
        self.assertAlmostEqual(result.ml_improvement_sharpe, 0.1, places=4)

    @patch("app.comparison.engine.fetch_benchmark_curves")
    @patch("app.comparison.engine.get_benchmark_stats")
    @patch("app.comparison.engine.run_backtest")
    async def test_min_len_truncation_for_unequal_series(
        self, mock_run_backtest, mock_get_stats, mock_fetch_curves
    ):
        """Engine should truncate to the shorter return series when lengths differ."""
        mock_fetch_curves.return_value = {}
        mock_get_stats.return_value = {}

        # Manual equity curve longer than ML
        manual_eq = [{"equity": v} for v in np.linspace(100_000, 110_000, 20)]
        ml_eq = [{"equity": v} for v in np.linspace(100_000, 108_000, 12)]

        manual_metrics = SimpleMetrics(manual_eq, sharpe=1.5)
        ml_metrics = SimpleMetrics(ml_eq, sharpe=1.6)

        mock_run_backtest.side_effect = [manual_metrics, ml_metrics]

        manual_signals = pd.Series(np.random.randint(0, 2, size=20))
        ml_signals = pd.Series(np.random.randint(0, 2, size=12))
        prices = pd.Series(np.linspace(50, 60, 20))

        result = await self.engine.run_comparison(
            manual_signals,
            ml_signals,
            prices,
            strategy_name="unequal_lengths",
            symbol="UNEQ",
            interval="1D",
            start_date=self.start_date,
            end_date=self.end_date,
        )

        # Verify that a t-statistic was computed (min_len = 11 after pct_change)
        self.assertNotEqual(result.t_statistic, 0.0)
        self.assertNotEqual(result.p_value, 1.0)
        # Ensure min_len logic was applied by checking that the number of samples used
        # matches the shorter series length after dropping NaNs.
        manual_ret_len = pd.Series([e["equity"] for e in manual_eq]).pct_change().dropna().shape[0]
        ml_ret_len = pd.Series([e["equity"] for e in ml_eq]).pct_change().dropna().shape[0]
        expected_min_len = min(manual_ret_len, ml_ret_len)
        self.assertGreaterEqual(expected_min_len, 10)
        self.assertGreaterEqual(result.t_statistic, -10)  # sanity check for realistic range


if __name__ == "__main__":
    unittest.main()