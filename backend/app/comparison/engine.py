"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from scipy import stats

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger

import unittest
import asyncio
from unittest.mock import patch, AsyncMock


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
        # Input validation
        if not isinstance(manual_signals, pd.Series):
            raise ValueError("manual_signals must be a pandas Series.")
        if not isinstance(ml_signals, pd.Series):
            raise ValueError("ml_signals must be a pandas Series.")
        if not isinstance(prices, pd.Series):
            raise ValueError("prices must be a pandas Series.")

        if manual_signals.empty:
            raise ValueError("manual_signals series cannot be empty.")
        if ml_signals.empty:
            raise ValueError("ml_signals series cannot be empty.")
        if prices.empty:
            raise ValueError("prices series cannot be empty.")

        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string.")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("interval must be a non-empty string.")

        if not isinstance(start_date, date):
            raise ValueError("start_date must be a datetime.date instance.")
        if not isinstance(end_date, date):
            raise ValueError("end_date must be a datetime.date instance.")
        if start_date > end_date:
            raise ValueError("start_date cannot be later than end_date.")

        if not isinstance(initial_equity, (int, float)):
            raise ValueError("initial_equity must be a numeric type.")
        if initial_equity <= 0:
            raise ValueError("initial_equity must be a positive number.")

        # Ensure series are aligned on the same index (optional but helps consistency)
        common_index = manual_signals.index.intersection(ml_signals.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError("manual_signals, ml_signals, and prices must share at least one common index.")
        manual_signals = manual_signals.loc[common_index]
        ml_signals = ml_signals.loc[common_index]
        prices = prices.loc[common_index]

        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

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
            is_significant=p_val < 0.05,
            winner=winner,
        )


# ----------------------------------------------------------------------
# Unit Tests for Edge Cases
# ----------------------------------------------------------------------
class TestStrategyComparisonEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = StrategyComparisonEngine()
        self.valid_date_start = date(2022, 1, 1)
        self.valid_date_end = date(2022, 1, 10)
        self.valid_series_index = pd.date_range(start="2022-01-01", periods=5, freq="D")
        self.valid_manual = pd.Series([1, 0, 1, 0, 1], index=self.valid_series_index)
        self.valid_ml = pd.Series([0, 1, 0, 1, 0], index=self.valid_series_index)
        self.valid_prices = pd.Series([100, 101, 102, 103, 104], index=self.valid_series_index)

        # Simple mock BacktestMetrics object
        class MockMetrics:
            def __init__(self, equity_curve, sharpe):
                self.equity_curve = equity_curve
                self.sharpe = sharpe

        self.mock_manual_metrics = MockMetrics(
            equity_curve=[{"equity": 100_000 + i * 1000} for i in range(5)],
            sharpe=1.0,
        )
        self.mock_ml_metrics = MockMetrics(
            equity_curve=[{"equity": 100_000 + i * 1500} for i in range(5)],
            sharpe=1.2,
        )

    async def test_empty_manual_signals_raises(self):
        empty_series = pd.Series([], dtype=float)
        with self.assertRaises(ValueError) as ctx:
            await self.engine.run_comparison(
                manual_signals=empty_series,
                ml_signals=self.valid_ml,
                prices=self.valid_prices,
                strategy_name="test",
                symbol="TEST",
                interval="1d",
                start_date=self.valid_date_start,
                end_date=self.valid_date_end,
            )
        self.assertIn("manual_signals series cannot be empty", str(ctx.exception))

    async def test_start_date_after_end_date_raises(self):
        later_start = date(2022, 2, 1)
        earlier_end = date(2022, 1, 1)
        with self.assertRaises(ValueError) as ctx:
            await self.engine.run_comparison(
                manual_signals=self.valid_manual,
                ml_signals=self.valid_ml,
                prices=self.valid_prices,
                strategy_name="test",
                symbol="TEST",
                interval="1d",
                start_date=later_start,
                end_date=earlier_end,
            )
        self.assertIn("start_date cannot be later than end_date", str(ctx.exception))

    async def test_no_common_index_raises(self):
        # Shift ml_signals index so there is no overlap
        shifted_ml = self.valid_ml.copy()
        shifted_ml.index = shifted_ml.index + pd.Timedelta(days=10)

        with self.assertRaises(ValueError) as ctx:
            await self.engine.run_comparison(
                manual_signals=self.valid_manual,
                ml_signals=shifted_ml,
                prices=self.valid_prices,
                strategy_name="test",
                symbol="TEST",
                interval="1d",
                start_date=self.valid_date_start,
                end_date=self.valid_date_end,
            )
        self.assertIn("must share at least one common index", str(ctx.exception))

    async def test_short_overlap_returns_default_statistics(self):
        # Use only 5 points (<=10) to trigger default t-statistic/p-value
        with patch("app.backtest.engine.run_backtest") as mock_run_backtest, \
             patch("app.comparison.benchmarks.fetch_benchmark_curves", new_callable=AsyncMock) as mock_fetch_curves, \
             patch("app.comparison.benchmarks.get_benchmark_stats") as mock_get_stats:

            mock_run_backtest.side_effect = [self.mock_manual_metrics, self.mock_ml_metrics]
            mock_fetch_curves.return_value = {}
            mock_get_stats.return_value = {}

            result = await self.engine.run_comparison(
                manual_signals=self.valid_manual,
                ml_signals=self.valid_ml,
                prices=self.valid_prices,
                strategy_name="test",
                symbol="TEST",
                interval="1d",
                start_date=self.valid_date_start,
                end_date=self.valid_date_end,
            )

            self.assertEqual(result.t_statistic, 0.0)
            self.assertEqual(result.p_value, 1.0)
            self.assertFalse(result.is_significant)
            # Verify winner logic respects improvement threshold
            self.assertIn(result.winner, {"ml", "neither"})


if __name__ == "__main__":
    unittest.main()