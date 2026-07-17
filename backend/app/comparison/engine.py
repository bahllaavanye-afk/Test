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


# ---------------------------------------------------------------------------
# Unit tests for edge cases
# ---------------------------------------------------------------------------
import unittest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta


class DummyMetrics:
    """Simple stand‑in for BacktestMetrics used in tests."""

    def __init__(self, sharpe: float, equity_curve: list[dict]):
        self.sharpe = sharpe
        self.equity_curve = equity_curve


class TestStrategyComparisonEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = StrategyComparisonEngine()
        self.start_date = date(2023, 1, 1)
        self.end_date = date(2023, 1, 31)
        # Minimal signal series – content irrelevant due to mocking
        self.manual_signals = pd.Series([1, 0, 1])
        self.ml_signals = pd.Series([0, 1, 0])
        self.prices = pd.Series([100, 101, 102])

    async def _run_engine(self, manual_metrics, ml_metrics, mock_ttest=None):
        """Helper to patch dependencies and execute run_comparison."""
        async_fetch = AsyncMock(return_value={"dummy": "benchmark"})
        with patch("app.backtest.engine.run_backtest") as mock_run_backtest, \
                patch("app.comparison.benchmarks.fetch_benchmark_curves", async_fetch), \
                patch("app.comparison.benchmarks.get_benchmark_stats", return_value={"dummy": "stats"}):
            mock_run_backtest.side_effect = [manual_metrics, ml_metrics]
            if mock_ttest:
                with patch.object(stats, "ttest_ind", mock_ttest):
                    result = await self.engine.run_comparison(
                        self.manual_signals,
                        self.ml_signals,
                        self.prices,
                        "test_strategy",
                        "TEST",
                        "1d",
                        self.start_date,
                        self.end_date,
                    )
            else:
                result = await self.engine.run_comparison(
                    self.manual_signals,
                    self.ml_signals,
                    self.prices,
                    "test_strategy",
                    "TEST",
                    "1d",
                    self.start_date,
                    self.end_date,
                )
        return result

    async def test_ttest_fallback_when_insufficient_data(self):
        """When return series length <= 10, t‑stat should be 0 and p‑value 1."""
        # Equity curve of length 5 → pct_change yields 4 points (<10)
        equity_curve = [{"equity": v} for v in [100, 105, 110, 115, 120]]
        manual_metrics = DummyMetrics(sharpe=1.0, equity_curve=equity_curve)
        ml_metrics = DummyMetrics(sharpe=1.2, equity_curve=equity_curve)

        result = await self._run_engine(manual_metrics, ml_metrics)

        self.assertEqual(result.t_statistic, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.is_significant)

    async def test_winner_neither_when_improvement_below_threshold(self):
        """If Sharpe improvement is less than 0.1, winner should be 'neither'."""
        # Use longer equity curve to bypass t‑test length check
        dates = pd.date_range(self.start_date, periods=15, freq="D")
        equity_curve = [{"equity": 100 + i} for i in range(15)]
        manual_metrics = DummyMetrics(sharpe=1.00, equity_curve=equity_curve)
        ml_metrics = DummyMetrics(sharpe=1.05, equity_curve=equity_curve)  # diff = 0.05

        result = await self._run_engine(manual_metrics, ml_metrics)

        self.assertAlmostEqual(result.ml_improvement_sharpe, 0.05, places=4)
        self.assertEqual(result.winner, "neither")

    async def test_significance_flag_when_pvalue_below_threshold(self):
        """If p‑value < 0.05, is_significant should be True and winner reflect Sharpe."""
        equity_curve = [{"equity": 100 + i} for i in range(20)]
        manual_metrics = DummyMetrics(sharpe=0.8, equity_curve=equity_curve)
        ml_metrics = DummyMetrics(sharpe=1.2, equity_curve=equity_curve)

        # Mock t‑test to return a low p‑value
        mock_ttest = lambda a, b: (0.5, 0.01)

        result = await self._run_engine(manual_metrics, ml_metrics, mock_ttest=mock_ttest)

        self.assertTrue(result.is_significant)
        self.assertAlmostEqual(result.p_value, 0.01, places=6)
        self.assertEqual(result.winner, "ml")


if __name__ == "__main__":
    unittest.main()