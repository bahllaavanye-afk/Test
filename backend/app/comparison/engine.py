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

# Constants
DEFAULT_INITIAL_EQUITY: float = 100_000.0
MIN_SAMPLE_SIZE: int = 10
IMPROVEMENT_THRESHOLD: float = 0.1
SIGNIFICANCE_LEVEL: float = 0.05
ROUND_SHARPE: int = 4
ROUND_PVALUE_LOG: int = 4
ROUND_PVALUE_RETURN: int = 6

ERROR_MANUAL_SIGNALS_TYPE = "manual_signals must be a pandas Series."
ERROR_ML_SIGNALS_TYPE = "ml_signals must be a pandas Series."
ERROR_PRICES_TYPE = "prices must be a pandas Series."
ERROR_MANUAL_SIGNALS_EMPTY = "manual_signals series cannot be empty."
ERROR_ML_SIGNALS_EMPTY = "ml_signals series cannot be empty."
ERROR_PRICES_EMPTY = "prices series cannot be empty."
ERROR_STRATEGY_NAME = "strategy_name must be a non-empty string."
ERROR_SYMBOL = "symbol must be a non-empty string."
ERROR_INTERVAL = "interval must be a non-empty string."
ERROR_START_DATE_TYPE = "start_date must be a datetime.date instance."
ERROR_END_DATE_TYPE = "end_date must be a datetime.date instance."
ERROR_DATE_ORDER = "start_date cannot be later than end_date."
ERROR_INITIAL_EQUITY_TYPE = "initial_equity must be a numeric type."
ERROR_INITIAL_EQUITY_POSITIVE = "initial_equity must be a positive number."
ERROR_COMMON_INDEX = "manual_signals, ml_signals, and prices must share at least one common index."

LOG_COMPARISON_COMPLETE = "Comparison complete"
WINNER_ML = "ml"
WINNER_MANUAL = "manual"
WINNER_NEITHER = "neither"


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
        # Input validation
        if not isinstance(manual_signals, pd.Series):
            raise ValueError(ERROR_MANUAL_SIGNALS_TYPE)
        if not isinstance(ml_signals, pd.Series):
            raise ValueError(ERROR_ML_SIGNALS_TYPE)
        if not isinstance(prices, pd.Series):
            raise ValueError(ERROR_PRICES_TYPE)

        if manual_signals.empty:
            raise ValueError(ERROR_MANUAL_SIGNALS_EMPTY)
        if ml_signals.empty:
            raise ValueError(ERROR_ML_SIGNALS_EMPTY)
        if prices.empty:
            raise ValueError(ERROR_PRICES_EMPTY)

        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError(ERROR_STRATEGY_NAME)
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(ERROR_SYMBOL)
        if not isinstance(interval, str) or not interval.strip():
            raise ValueError(ERROR_INTERVAL)

        if not isinstance(start_date, date):
            raise ValueError(ERROR_START_DATE_TYPE)
        if not isinstance(end_date, date):
            raise ValueError(ERROR_END_DATE_TYPE)
        if start_date > end_date:
            raise ValueError(ERROR_DATE_ORDER)

        if not isinstance(initial_equity, (int, float)):
            raise ValueError(ERROR_INITIAL_EQUITY_TYPE)
        if initial_equity <= 0:
            raise ValueError(ERROR_INITIAL_EQUITY_POSITIVE)

        # Ensure series are aligned on the same index (optional but helps consistency)
        common_index = manual_signals.index.intersection(ml_signals.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError(ERROR_COMMON_INDEX)
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
        if min_len > MIN_SAMPLE_SIZE:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
        else:
            t_stat, p_val = 0.0, 1.0

        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = WINNER_ML if ml_metrics.sharpe > manual_metrics.sharpe else WINNER_MANUAL
        if abs(improvement) < IMPROVEMENT_THRESHOLD:
            winner = WINNER_NEITHER

        logger.info(
            LOG_COMPARISON_COMPLETE,
            strategy=strategy_name,
            manual_sharpe=manual_metrics.sharpe,
            ml_sharpe=ml_metrics.sharpe,
            p_value=round(p_val, ROUND_PVALUE_LOG),
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
            ml_improvement_sharpe=round(improvement, ROUND_SHARPE),
            t_statistic=round(float(t_stat), ROUND_SHARPE),
            p_value=round(float(p_val), ROUND_PVALUE_RETURN),
            is_significant=p_val < SIGNIFICANCE_LEVEL,
            winner=winner,
        )