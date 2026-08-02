"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Tuple

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
    # Simple in‑memory caches to avoid repeated heavy work
    _benchmark_cache: dict[Tuple[date, date], dict] = {}
    _backtest_cache: dict[Tuple[Tuple[Any, ...], Tuple[Any, ...]], BacktestMetrics] = {}

    @staticmethod
    def _hash_series(series: pd.Series) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """
        Produce a hashable representation of a pandas Series based on its values and index.
        """
        # Convert values and index to immutable tuples; this is cheap for typical series sizes
        return tuple(series.values.tolist()), tuple(series.index.astype(str).tolist())

    def _run_backtest_cached(self, signals: pd.Series, prices: pd.Series, initial_equity: float) -> BacktestMetrics:
        """
        Run backtest with memoization to skip duplicate calculations.
        """
        key = (self._hash_series(signals), self._hash_series(prices), initial_equity)
        if key in self._backtest_cache:
            logger.debug("Backtest cache hit")
            return self._backtest_cache[key]

        logger.debug("Backtest cache miss – executing run_backtest")
        result = run_backtest(signals, prices, initial_equity)
        self._backtest_cache[key] = result
        return result

    async def _fetch_benchmark_curves_cached(self, start_date: date, end_date: date) -> dict:
        """
        Cached async fetch for benchmark curves to avoid repeated network calls.
        """
        cache_key = (start_date, end_date)
        if cache_key in self._benchmark_cache:
            logger.debug("Benchmark cache hit for %s - %s", start_date, end_date)
            return self._benchmark_cache[cache_key]

        logger.debug("Benchmark cache miss – fetching curves")
        curves = await fetch_benchmark_curves(start_date, end_date)
        self._benchmark_cache[cache_key] = curves
        return curves

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

        # Align series on common index
        common_index = manual_signals.index.intersection(ml_signals.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError("manual_signals, ml_signals, and prices must share at least one common index.")
        manual_signals = manual_signals.loc[common_index]
        ml_signals = ml_signals.loc[common_index]
        prices = prices.loc[common_index]

        # Run backtests with caching
        manual_metrics = self._run_backtest_cached(manual_signals, prices, initial_equity)
        ml_metrics = self._run_backtest_cached(ml_signals, prices, initial_equity)

        # Fetch benchmark data with caching
        benchmark_curves = await self._fetch_benchmark_curves_cached(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Vectorized extraction of equity curves
        manual_eq = pd.Series(pd.DataFrame(manual_metrics.equity_curve)["equity"])
        ml_eq = pd.Series(pd.DataFrame(ml_metrics.equity_curve)["equity"])

        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()

        # Early exit for very short series
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