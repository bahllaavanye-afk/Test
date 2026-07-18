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
# Unit Tests for Edge Cases
# ==============================
import unittest
from unittest.mock import patch, MagicMock


class MockMetrics:
    """Simple mock mimicking BacktestMetrics with required attributes."""
    def __init__(self, equity_curve, sharpe):
        self.equity_curve = equity_curve
        self.sharpe = sharpe


class TestStrategyComparisonEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = StrategyComparisonEngine()
        # Minimal series to satisfy function signatures
        self.signals = pd.Series([1, 0, 1])
        self.prices = pd.Series([100, 101, 102])

    @patch("app.comparison.engine.fetch_benchmark_curves")
    @patch("app.comparison.engine.get_benchmark_stats")
    @patch("app.comparison.engine.run_backtest")
    async def test_ttest_fallback_when_series_too_short(
        self, mock_run_backtest, mock_get_stats, mock_fetch_curves
    ):
        """When return series length <=10, engine should fallback to t_stat=0.0, p_val=1.0."""
        # Create equity curves with only 5 points each
        equity_curve = [{"equity": v} for v in [100_000, 100_500, 101_000, 101_500, 102_000]]
        mock_metrics = MockMetrics(equity_curve, sharpe=1.0)
        mock_run_backtest.return_value = mock_metrics

        mock_fetch_curves.return_value = {}
        mock_get_stats.return_value = {}

        result = await self.engine.run_comparison(
            manual_signals=self.signals,
            ml_signals=self.signals,
            prices=self.prices,
            strategy_name="test_strategy",
            symbol="TEST",
            interval="1d",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
        )

        self.assertEqual(result.t_statistic, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.is_significant)

    @patch("app.comparison.engine.fetch_benchmark_curves")
    @patch("app.comparison.engine.get_benchmark_stats")
    @patch("app.comparison.engine.run_backtest")
    async def test_winner_neither_when_sharpe_difference_small(
        self, mock_run_backtest, mock_get_stats, mock_fetch_curves
    ):
        """If Sharpe difference is less than 0.1, winner should be 'neither'."""
        equity_curve = [{"equity": v} for v in np.linspace(100_000, 110_000, 15)]
        manual_metrics = MockMetrics(equity_curve, sharpe=1.00)
        ml_metrics = MockMetrics(equity_curve, sharpe=1.05)  # diff = 0.05 < 0.1
        # run_backtest called twice: first for manual, then for ml
        mock_run_backtest.side_effect = [manual_metrics, ml_metrics]

        mock_fetch_curves.return_value = {}
        mock_get_stats.return_value = {}

        result = await self.engine.run_comparison(
            manual_signals=self.signals,
            ml_signals=self.signals,
            prices=self.prices,
            strategy_name="edge_case_strategy",
            symbol="EDG",
            interval="1d",
            start_date=date(2023, 2, 1),
            end_date=date(2023, 2, 20),
        )

        self.assertEqual(result.winner, "neither")
        self.assertAlmostEqual(result.ml_improvement_sharpe, 0.05, places=4)

    @patch("app.comparison.engine.fetch_benchmark_curves")
    @patch("app.comparison.engine.get_benchmark_stats")
    @patch("app.comparison.engine.run_backtest")
    async def test_winner_ml_when_sharpe_significantly_higher(
        self, mock_run_backtest, mock_get_stats, mock_fetch_curves
    ):
        """When ML Sharpe exceeds manual by >=0.1, winner should be 'ml'."""
        equity_curve = [{"equity": v} for v in np.linspace(100_000, 120_000, 30)]
        manual_metrics = MockMetrics(equity_curve, sharpe=0.90)
        ml_metrics = MockMetrics(equity_curve, sharpe=1.20)  # diff = 0.30
        mock_run_backtest.side_effect = [manual_metrics, ml_metrics]

        mock_fetch_curves.return_value = {}
        mock_get_stats.return_value = {}

        result = await self.engine.run_comparison(
            manual_signals=self.signals,
            ml_signals=self.signals,
            prices=self.prices,
            strategy_name="ml_advantage",
            symbol="MLA",
            interval="1d",
            start_date=date(2023, 3, 1),
            end_date=date(2023, 3, 31),
        )

        self.assertEqual(result.winner, "ml")
        self.assertGreater(result.ml_improvement_sharpe, 0.1)
        # Ensure statistical significance flag reflects p-value
        self.assertIsInstance(result.is_significant, bool)

if __name__ == "__main__":
    unittest.main()