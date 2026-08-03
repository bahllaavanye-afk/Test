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
DEFAULT_INITIAL_EQUITY: float = 100_000
MIN_SAMPLE_SIZE: int = 10
IMPROVEMENT_THRESHOLD: float = 0.1
SIGNIFICANCE_LEVEL: float = 0.05
P_VALUE_ROUND: int = 4
IMPROVEMENT_ROUND: int = 4
T_STAT_ROUND: int = 4
P_VAL_FINAL_ROUND: int = 6

# Log field names
LOG_STRATEGY_FIELD: str = "strategy"
LOG_MANUAL_SHARPE_FIELD: str = "manual_sharpe"
LOG_ML_SHARPE_FIELD: str = "ml_sharpe"
LOG_P_VALUE_FIELD: str = "p_value"
LOG_MESSAGE_COMPARISON_COMPLETE: str = "Comparison complete"
LOG_ERROR_FETCH_BENCHMARK: str = "Failed to fetch benchmark curves"
LOG_ERROR_KEY: str = "error"

# Winner identifiers
WINNER_ML: str = "ml"
WINNER_MANUAL: str = "manual"
WINNER_NEITHER: str = "neither"


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
        initial_equity: float = DEFAULT_INITIAL_EQUITY,
    ) -> ComparisonResult:
        """Run a side‑by‑side backtest comparison between manual and ML‑enhanced signals.

        Validates inputs, aligns series, executes backtests, fetches benchmark data,
        and computes statistical significance of the Sharpe improvement.
        """
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

        # Align series on common index
        common_index = manual_signals.index.intersection(ml_signals.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError("manual_signals, ml_signals, and prices must share at least one common index.")
        manual_signals = manual_signals.loc[common_index]
        ml_signals = ml_signals.loc[common_index]
        prices = prices.loc[common_index]

        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        # Fetch benchmark data safely
        try:
            benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        except Exception as exc:
            logger.error(LOG_ERROR_FETCH_BENCHMARK, **{LOG_ERROR_KEY: str(exc)})
            benchmark_curves = {}
        benchmark_stats = get_benchmark_stats()

        # Equity curve processing
        manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()

        # Statistical test
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
            LOG_MESSAGE_COMPARISON_COMPLETE,
            **{
                LOG_STRATEGY_FIELD: strategy_name,
                LOG_MANUAL_SHARPE_FIELD: manual_metrics.sharpe,
                LOG_ML_SHARPE_FIELD: ml_metrics.sharpe,
                LOG_P_VALUE_FIELD: round(p_val, P_VALUE_ROUND),
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