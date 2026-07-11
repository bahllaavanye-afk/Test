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


# ==================== Unit Tests ====================

import unittest
from unittest.mock import patch, MagicMock


class DummyMetrics:
    """Simple stand‑in for BacktestMetrics used in tests."""
    def __init__(self, equity_curve, sharpe):
        self.equity_curve = equity_curve
        self.sharpe = sharpe


class TestStrategyComparisonEngine(unittest.IsolatedAsyncioTestCase):
    async def _run_engine(
        self,
        manual_eq,
        ml_eq,
        manual_sharpe,
        ml_sharpe,
        mock_ttest=None,
    ):
        manual_signals = pd.Series([1, 0, 1])
        ml_signals = pd.Series([1, 1, 0])
        prices = pd.Series([100, 101, 102])

        manual_metrics = DummyMetrics(
            equity_curve=[{"equity": v} for v in manual_eq],
            sharpe=manual_sharpe,
        )
        ml_metrics = DummyMetrics(
            equity_curve=[{"equity": v} for v in ml_eq],
            sharpe=ml_sharpe,
        )

        patches = [
            patch("backend.app.comparison.engine.run_backtest", side_effect=[manual_metrics, ml_metrics]),
            patch("backend.app.comparison.engine.fetch_benchmark_curves", return_value={}),
            patch("backend.app.comparison.engine.get_benchmark_stats", return_value={}),
        ]

        if mock_ttest is not None:
            patches.append(patch("backend.app.comparison.engine.stats.ttest_ind", mock_ttest))

        async with asyncio.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            engine = StrategyComparisonEngine()
            result = await engine.run_comparison(
                manual_signals,
                ml_signals,
                prices,
                strategy_name="test_strategy",
                symbol="TEST",
                interval="1d",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 1, 10),
            )
        return result

    async def test_t_stat_boundary_short_series(self):
        """When return series length <= 10, t‑stat should be 0 and p‑value 1."""
        result = await self._run_engine(
            manual_eq=[100, 101, 102, 103, 104, 105],
            ml_eq=[100, 102, 104, 106, 108, 110],
            manual_sharpe=1.0,
            ml_sharpe=1.2,
        )
        self.assertEqual(result.t_statistic, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.is_significant)

    async def test_winner_neither_due_to_small_improvement(self):
        """If Sharpe improvement is below 0.1, winner should be 'neither'."""
        result = await self._run_engine(
            manual_eq=[100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150],
            ml_eq=[100, 106, 112, 118, 124, 130, 136, 142, 148, 154, 160],
            manual_sharpe=1.00,
            ml_sharpe=1.05,  # improvement = 0.05 < 0.1
        )
        self.assertEqual(result.winner, "neither")
        self.assertAlmostEqual(result.ml_improvement_sharpe, 0.05, places=4)

    async def test_significance_flag_true(self):
        """When the mocked p‑value is below 0.05, is_significant should be True."""
        mock_ttest = MagicMock(return_value=(2.5, 0.03))
        result = await self._run_engine(
            manual_eq=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            ml_eq=[100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122],
            manual_sharpe=0.8,
            ml_sharpe=1.0,
            mock_ttest=mock_ttest,
        )
        self.assertTrue(result.is_significant)
        self.assertAlmostEqual(result.p_value, 0.03, places=6)
        self.assertAlmostEqual(result.t_statistic, 2.5, places=4)


if __name__ == "__main__":
    unittest.main()