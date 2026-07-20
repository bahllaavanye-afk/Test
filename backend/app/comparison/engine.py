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
    @staticmethod
    def _validate_series(name: str, series: pd.Series) -> None:
        if not isinstance(series, pd.Series):
            raise ValueError(f"'{name}' must be a pandas Series.")
        if series.empty:
            raise ValueError(f"'{name}' cannot be empty.")
        if not np.issubdtype(series.dtype, np.number):
            raise ValueError(f"'{name}' must contain numeric values.")

    @staticmethod
    def _validate_string(name: str, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError(f"'{name}' must be a string.")
        if not value.strip():
            raise ValueError(f"'{name}' cannot be empty or whitespace.")

    @staticmethod
    def _validate_date(name: str, value: date) -> None:
        if not isinstance(value, date):
            raise ValueError(f"'{name}' must be a datetime.date instance.")

    @staticmethod
    def _validate_initial_equity(value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("initial_equity must be a numeric type.")
        if value <= 0:
            raise ValueError("initial_equity must be positive.")

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
        self._validate_series("manual_signals", manual_signals)
        self._validate_series("ml_signals", ml_signals)
        self._validate_series("prices", prices)

        self._validate_string("strategy_name", strategy_name)
        self._validate_string("symbol", symbol)
        self._validate_string("interval", interval)

        self._validate_date("start_date", start_date)
        self._validate_date("end_date", end_date)
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date.")

        self._validate_initial_equity(initial_equity)

        # Ensure the series align on the same index for fair comparison
        common_index = manual_signals.index.intersection(ml_signals.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError("No overlapping index between manual_signals, ml_signals, and prices.")
        manual_signals = manual_signals.loc[common_index]
        ml_signals = ml_signals.loc[common_index]
        prices = prices.loc[common_index]

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