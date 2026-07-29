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
import unittest
import asyncio
from unittest.mock import patch, AsyncMock


class _MockMetrics:
    """Simple mock for BacktestMetrics with required attributes."""
    def __init__(self, equity_curve, sharpe):
        self.equity_curve = equity_curve
        self.sharpe = sharpe


class TestStrategyComparisonEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = StrategyComparisonEngine()
        self.base_date = date(2023, 1, 1)
        self.end_date = date(2023, 1, 10)

        # Common index for successful path
        self.idx = pd.date_range(start="2023-01-01", periods=15, freq="D")
        self.manual_signals = pd.Series([1] * 15, index=self.idx)
        self.ml_signals = pd.Series([1] * 15, index=self.idx)
        self.prices = pd.Series([100 + i for i in range(15)], index=self.idx)

        # Minimal equity curve mock (equity values increasing linearly)
        self.equity_curve = [{"equity": 100_000 + i * 1000} for i in range(15)]

    @patch("backend.app.comparison.engine.run_backtest")
    @patch("backend.app.comparison.engine.fetch_benchmark_curves", new_callable=AsyncMock)
    @patch("backend.app.comparison.engine.get_benchmark_stats")
    async def test_empty_manual_signals_raises(self, mock_bench_stats, mock_fetch_curves, mock_run_bt):
        mock_run_bt.return_value = _MockMetrics(self.equity_curve, 1.0)
        mock_fetch_curves.return_value = {}
        mock_bench_stats.return_value = {}

        empty_series = pd.Series([], dtype=float)

        with self.assertRaises(ValueError) as cm:
            await self.engine.run_comparison(
                manual_signals=empty_series,
                ml_signals=self.ml_signals,
                prices=self.prices,
                strategy_name="TestStrategy",
                symbol="TEST",
                interval="1D",
                start_date=self.base_date,
                end_date=self.end_date,
            )
        self.assertIn("manual_signals series cannot be empty", str(cm.exception))

    @patch("backend.app.comparison.engine.run_backtest")
    @patch("backend.app.comparison.engine.fetch_benchmark_curves", new_callable=AsyncMock)
    @patch("backend.app.comparison.engine.get_benchmark_stats")
    async def test_start_date_after_end_date_raises(self, mock_bench_stats, mock_fetch_curves, mock_run_bt):
        mock_run_bt.return_value = _MockMetrics(self.equity_curve, 1.0)
        mock_fetch_curves.return_value = {}
        mock_bench_stats.return_value = {}

        with self.assertRaises(ValueError) as cm:
            await self.engine.run_comparison(
                manual_signals=self.manual_signals,
                ml_signals=self.ml_signals,
                prices=self.prices,
                strategy_name="TestStrategy",
                symbol="TEST",
                interval="1D",
                start_date=self.end_date,
                end_date=self.base_date,
            )
        self.assertIn("start_date cannot be later than end_date", str(cm.exception))

    @patch("backend.app.comparison.engine.run_backtest")
    @patch("backend.app.comparison.engine.fetch_benchmark_curves", new_callable=AsyncMock)
    @patch("backend.app.comparison.engine.get_benchmark_stats")
    async def test_insufficient_common_index_uses_default_statistics(self, mock_bench_stats, mock_fetch_curves, mock_run_bt):
        # Provide non‑overlapping indices to trigger the common index empty error
        manual = pd.Series([1, 1], index=pd.date_range("2023-01-01", periods=2))
        ml = pd.Series([1, 1], index=pd.date_range("2023-02-01", periods=2))
        prices = pd.Series([100, 101], index=pd.date_range("2023-03-01", periods=2))

        mock_run_bt.return_value = _MockMetrics(self.equity_curve, 1.0)
        mock_fetch_curves.return_value = {}
        mock_bench_stats.return_value = {}

        with self.assertRaises(ValueError) as cm:
            await self.engine.run_comparison(
                manual_signals=manual,
                ml_signals=ml,
                prices=prices,
                strategy_name="TestStrategy",
                symbol="TEST",
                interval="1D",
                start_date=self.base_date,
                end_date=self.end_date,
            )
        self.assertIn("must share at least one common index", str(cm.exception))

    @patch("backend.app.comparison.engine.run_backtest")
    @patch("backend.app.comparison.engine.fetch_benchmark_curves", new_callable=AsyncMock)
    @patch("backend.app.comparison.engine.get_benchmark_stats")
    async def test_boundary_min_len_returns_default_t_and_p(self, mock_bench_stats, mock_fetch_curves, mock_run_bt):
        # Create series where after pct_change we have only 5 returns (<10)
        short_idx = pd.date_range(start="2023-01-01", periods=6, freq="D")
        manual = pd.Series([1] * 6, index=short_idx)
        ml = pd.Series([1] * 6, index=short_idx)
        prices = pd.Series([100 + i for i in range(6)], index=short_idx)

        mock_run_bt.return_value = _MockMetrics(
            [{"equity": 100_000 + i * 1000} for i in range(6)], sharpe=1.0
        )
        mock_fetch_curves.return_value = {}
        mock_bench_stats.return_value = {}

        result = await self.engine.run_comparison(
            manual_signals=manual,
            ml_signals=ml,
            prices=prices,
            strategy_name="BoundaryStrategy",
            symbol="BND",
            interval="1D",
            start_date=self.base_date,
            end_date=self.end_date,
        )

        self.assertEqual(result.t_statistic, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.is_significant)


if __name__ == "__main__":
    unittest.main()