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


# -------------------------------------------------------------------------
# Unit Tests for Edge Cases
# -------------------------------------------------------------------------
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import datetime
import asyncio


class TestStrategyComparisonEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Minimal viable BacktestMetrics mock
        self.mock_metrics = MagicMock(spec=BacktestMetrics)
        self.mock_metrics.sharpe = 1.0
        self.mock_metrics.equity_curve = [{"equity": 100_000}, {"equity": 101_000}, {"equity": 102_500}]

    async def test_empty_manual_signals_raises(self):
        engine = StrategyComparisonEngine()
        empty_series = pd.Series(dtype=float)
        ml_series = pd.Series([1, 0, 1], index=[0, 1, 2])
        price_series = pd.Series([100, 101, 102], index=[0, 1, 2])

        with self.assertRaises(ValueError) as ctx:
            await engine.run_comparison(
                manual_signals=empty_series,
                ml_signals=ml_series,
                prices=price_series,
                strategy_name="test",
                symbol="TEST",
                interval="1d",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 1, 31),
            )
        self.assertIn("manual_signals series cannot be empty", str(ctx.exception))

    async def test_start_date_after_end_date_raises(self):
        engine = StrategyComparisonEngine()
        series = pd.Series([1, 0, 1], index=[0, 1, 2])
        with self.assertRaises(ValueError) as ctx:
            await engine.run_comparison(
                manual_signals=series,
                ml_signals=series,
                prices=series,
                strategy_name="test",
                symbol="TEST",
                interval="1d",
                start_date=date(2023, 2, 1),
                end_date=date(2023, 1, 31),
            )
        self.assertIn("start_date cannot be later than end_date", str(ctx.exception))

    async def test_insufficient_common_index_raises(self):
        engine = StrategyComparisonEngine()
        manual = pd.Series([1, 0], index=[0, 1])
        ml = pd.Series([1, 0], index=[2, 3])  # no overlap
        prices = pd.Series([100, 101], index=[0, 1])

        with self.assertRaises(ValueError) as ctx:
            await engine.run_comparison(
                manual_signals=manual,
                ml_signals=ml,
                prices=prices,
                strategy_name="test",
                symbol="TEST",
                interval="1d",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 1, 31),
            )
        self.assertIn("must share at least one common index", str(ctx.exception))

    async def test_t_statistic_fallback_when_data_short(self):
        engine = StrategyComparisonEngine()

        # Create short series (5 points) to trigger fallback
        idx = pd.date_range(start="2023-01-01", periods=5, freq="D")
        manual = pd.Series([1, 0, 1, 0, 1], index=idx)
        ml = pd.Series([0, 1, 0, 1, 0], index=idx)
        prices = pd.Series([100, 101, 102, 103, 104], index=idx)

        # Patch backtest and benchmark calls
        with patch("app.backtest.engine.run_backtest", return_value=self.mock_metrics), \
             patch("app.comparison.benchmarks.fetch_benchmark_curves", new_callable=AsyncMock) as mock_fetch, \
             patch("app.comparison.benchmarks.get_benchmark_stats", return_value={"dummy": 0}):

            mock_fetch.return_value = {"benchmark": []}
            result = await engine.run_comparison(
                manual_signals=manual,
                ml_signals=ml,
                prices=prices,
                strategy_name="short_test",
                symbol="SHORT",
                interval="1d",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 1, 31),
            )

        self.assertEqual(result.t_statistic, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.is_significant)


if __name__ == "__main__":
    unittest.main()