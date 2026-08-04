"""
Strategy Comparison Engine: run manual vs ML‑enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import numbers
import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from scipy import stats

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger

# ==============================
# Configuration Constants
# ==============================
DEFAULT_INITIAL_EQUITY: float = 100_000
MIN_SAMPLE_SIZE: int = 10
IMPROVEMENT_THRESHOLD: float = 0.1
SIGNIFICANCE_LEVEL: float = 0.05
P_VALUE_ROUND: int = 4
IMPROVEMENT_ROUND: int = 4
T_STAT_ROUND: int = 4
P_VAL_FINAL_ROUND: int = 6

# Default statistical values
DEFAULT_T_STAT: float = 0.0
DEFAULT_P_VAL: float = 1.0

# Log field names
LOG_STRATEGY_FIELD: str = "strategy"
LOG_MANUAL_SHARPE_FIELD: str = "manual_sharpe"
LOG_ML_SHARPE_FIELD: str = "ml_sharpe"
LOG_P_VALUE_FIELD: str = "p_value"
LOG_MANUAL_SIGNAL_COUNT: str = "manual_signal_count"
LOG_ML_SIGNAL_COUNT: str = "ml_signal_count"
LOG_EXECUTION_TIME: str = "execution_time_seconds"
LOG_MANUAL_PNL: str = "manual_pnl"
LOG_ML_PNL: str = "ml_pnl"

# Additional constants for string literals
LOG_COMPARISON_COMPLETE_MSG: str = "Comparison complete"
ERROR_FETCH_BENCHMARK_MSG: str = "Failed to fetch benchmark curves"
ERROR_COMMON_INDEX_MSG: str = "manual_signals, ml_signals, and prices must share at least one common index."
ERROR_DATE_ORDER_MSG: str = "start_date cannot be later than end_date."
ERROR_INITIAL_EQUITY_TYPE_MSG: str = "initial_equity must be a numeric type."
ERROR_INITIAL_EQUITY_POSITIVE_MSG: str = "initial_equity must be a positive number."
EQUITY_KEY: str = "equity"
WINNER_ML: str = "ml"
WINNER_MANUAL: str = "manual"
WINNER_NEITHER: str = "neither"

# Validation messages
MSG_MUST_BE_PANDAS_SERIES: str = "must be a pandas Series."
MSG_SERIES_CANNOT_BE_EMPTY: str = "series cannot be empty."
MSG_SERIES_CONTAINS_NAN: str = "series contains NaN values."
MSG_INDEX_MUST_BE_MONOTONIC: str = "index must be monotonic increasing."
MSG_SERIES_MUST_BE_NUMERIC: str = "series must contain numeric values."
MSG_STRING_NON_EMPTY: str = "must be a non‑empty string."
MSG_DATE_TYPE: str = "must be a datetime.date instance."

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
    winner: str = WINNER_NEITHER


class StrategyComparisonEngine:
    @staticmethod
    def _validate_series(series: pd.Series, name: str) -> None:
        """Validate that a pandas Series is suitable for backtesting."""
        if not isinstance(series, pd.Series):
            raise ValueError(f"{name} {MSG_MUST_BE_PANDAS_SERIES}")
        if series.empty:
            raise ValueError(f"{name} {MSG_SERIES_CANNOT_BE_EMPTY}")
        if series.isnull().any():
            raise ValueError(f"{name} {MSG_SERIES_CONTAINS_NAN}")
        if not series.index.is_monotonic_increasing:
            raise ValueError(f"{name} {MSG_INDEX_MUST_BE_MONOTONIC}")
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"{name} {MSG_SERIES_MUST_BE_NUMERIC}")

    @staticmethod
    def _validate_string(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} {MSG_STRING_NON_EMPTY}")

    @staticmethod
    def _validate_date(value: date, name: str) -> None:
        if not isinstance(value, date):
            raise ValueError(f"{name} {MSG_DATE_TYPE}")

    @staticmethod
    def _validate_initial_equity(value: numbers.Number) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError(ERROR_INITIAL_EQUITY_TYPE_MSG)
        if value <= 0:
            raise ValueError(ERROR_INITIAL_EQUITY_POSITIVE_MSG)

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
        initial_equity: float = DEFAULT_INITIAL_EQUITY,
    ) -> ComparisonResult:
        """Run a side‑by‑side backtest comparison between manual and ML‑enhanced signals.

        Validates inputs, aligns series, executes backtests, fetches benchmark data,
        and computes statistical significance of the Sharpe improvement.
        """
        # Input validation
        self._validate_series(manual_signals, "manual_signals")
        self._validate_series(ml_signals, "ml_signals")
        self._validate_series(prices, "prices")
        self._validate_string(strategy_name, "strategy_name")
        self._validate_string(symbol, "symbol")
        self._validate_string(interval, "interval")
        self._validate_date(start_date, "start_date")
        self._validate_date(end_date, "end_date")
        if start_date > end_date:
            raise ValueError(ERROR_DATE_ORDER_MSG)
        self._validate_initial_equity(initial_equity)

        # Start timing for monitoring
        start_time = time.perf_counter()

        # Align series on common index
        common_index = manual_signals.index.intersection(ml_signals.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError(ERROR_COMMON_INDEX_MSG)
        manual_signals = manual_signals.loc[common_index]
        ml_signals = ml_signals.loc[common_index]
        prices = prices.loc[common_index]

        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        # Fetch benchmark data safely
        try:
            benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        except Exception as exc:
            logger.error(ERROR_FETCH_BENCHMARK_MSG, error=str(exc))
            benchmark_curves = {}
        benchmark_stats = get_benchmark_stats()

        # Equity curve processing
        manual_eq = pd.Series([e[EQUITY_KEY] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e[EQUITY_KEY] for e in ml_metrics.equity_curve])
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()

        # Statistical test
        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > MIN_SAMPLE_SIZE:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
        else:
            t_stat, p_val = DEFAULT_T_STAT, DEFAULT_P_VAL

        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = WINNER_ML if ml_metrics.sharpe > manual_metrics.sharpe else WINNER_MANUAL
        if abs(improvement) < IMPROVEMENT_THRESHOLD:
            winner = WINNER_NEITHER

        # Compute monitoring metrics
        execution_time = time.perf_counter() - start_time
        manual_pnl = manual_eq.iloc[-1] - initial_equity if not manual_eq.empty else 0.0
        ml_pnl = ml_eq.iloc[-1] - initial_equity if not ml_eq.empty else 0.0

        logger.info(
            LOG_COMPARISON_COMPLETE_MSG,
            **{
                LOG_STRATEGY_FIELD: strategy_name,
                LOG_MANUAL_SHARPE_FIELD: manual_metrics.sharpe,
                LOG_ML_SHARPE_FIELD: ml_metrics.sharpe,
                LOG_P_VALUE_FIELD: round(p_val, P_VALUE_ROUND),
                LOG_MANUAL_SIGNAL_COUNT: len(manual_signals),
                LOG_ML_SIGNAL_COUNT: len(ml_signals),
                LOG_EXECUTION_TIME: round(execution_time, 4),
                LOG_MANUAL_PNL: round(manual_pnl, 2),
                LOG_ML_PNL: round(ml_pnl, 2),
            },
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
            ml_improvement_sharpe=round(improvement, IMPROVEMENT_ROUND),
            t_statistic=round(float(t_stat), T_STAT_ROUND),
            p_value=round(float(p_val), P_VAL_FINAL_ROUND),
            is_significant=p_val < SIGNIFICANCE_LEVEL,
            winner=winner,
        )