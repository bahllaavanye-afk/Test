"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Tuple

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
        """Run a full comparison between a manual and an ML‑enhanced strategy.

        The method validates inputs, aligns series, runs backtests, fetches benchmark
        data, computes statistical significance and returns a populated
        :class:`ComparisonResult`.
        """
        self._validate_inputs(
            manual_signals,
            ml_signals,
            prices,
            strategy_name,
            symbol,
            interval,
            start_date,
            end_date,
            initial_equity,
        )

        manual_signals, ml_signals, prices = self._align_series(
            manual_signals, ml_signals, prices
        )

        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        manual_ret, ml_ret = self._compute_returns(manual_metrics, ml_metrics)

        t_stat, p_val = self._perform_stat_test(manual_ret, ml_ret)

        improvement, winner = self._determine_winner(manual_metrics, ml_metrics)

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

    @staticmethod
    def _validate_inputs(
        manual_signals: pd.Series,
        ml_signals: pd.Series,
        prices: pd.Series,
        strategy_name: str,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
        initial_equity: float,
    ) -> None:
        """Validate all inputs to ``run_comparison``."""
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
            raise ValueError("strategy_name must be a non‑empty string.")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non‑empty string.")
        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("interval must be a non‑empty string.")

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

    @staticmethod
    def _align_series(
        manual: pd.Series, ml: pd.Series, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Align the three series on their common index."""
        common_index = manual.index.intersection(ml.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError(
                "manual_signals, ml_signals, and prices must share at least one common index."
            )
        return (
            manual.loc[common_index],
            ml.loc[common_index],
            prices.loc[common_index],
        )

    @staticmethod
    def _compute_returns(
        manual_metrics: BacktestMetrics, ml_metrics: BacktestMetrics
    ) -> Tuple[pd.Series, pd.Series]:
        """Extract equity curves and compute percentage returns."""
        manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()
        return manual_ret, ml_ret

    @staticmethod
    def _perform_stat_test(
        manual_ret: pd.Series, ml_ret: pd.Series
    ) -> Tuple[float, float]:
        """Run an independent‑samples t‑test if enough data points are available."""
        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
        else:
            t_stat, p_val = 0.0, 1.0
        return t_stat, p_val

    @staticmethod
    def _determine_winner(
        manual_metrics: BacktestMetrics, ml_metrics: BacktestMetrics
    ) -> Tuple[float, str]:
        """Calculate Sharpe improvement and decide the winner."""
        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = "ml" if ml_metrics.sharpe > manual_metrics.sharpe else "manual"
        if abs(improvement) < 0.1:
            winner = "neither"
        return improvement, winner