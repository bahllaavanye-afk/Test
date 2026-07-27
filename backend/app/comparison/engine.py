"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

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
    """
    Engine that evaluates a manual signal series against an ML‑enhanced counterpart.
    It applies lightweight confirmation filters to tighten entry conditions before
    delegating to the back‑test engine.
    """

    def _apply_entry_filters(
        self,
        signals: pd.Series,
        prices: pd.Series,
        persist_window: int = 2,
        price_move_window: int = 5,
        price_move_threshold: float = 0.001,
    ) -> pd.Series:
        """
        Tighten entry conditions by:
        1. Requiring the signal to be unchanged for ``persist_window`` consecutive periods.
        2. Ensuring the price moves at least ``price_move_threshold`` (relative) within the next
           ``price_move_window`` periods.

        Parameters
        ----------
        signals: pd.Series
            Original signal series (1 for entry, 0/NaN for no entry).
        prices: pd.Series
            Corresponding price series.
        persist_window: int
            Minimum number of consecutive identical signals required.
        price_move_window: int
            Look‑ahead window to evaluate price movement.
        price_move_threshold: float
            Minimum relative price change (e.g., 0.001 = 0.1 %).

        Returns
        -------
        pd.Series
            Filtered signal series with the same index; entries not meeting criteria are set to 0.
        """
        # Ensure boolean signals for easier handling
        sig_bool = signals.fillna(0).astype(bool)

        # 1. Persistence filter
        persistence = sig_bool.rolling(window=persist_window, min_periods=persist_window).apply(
            lambda x: x.all(), raw=True
        ).astype(bool)

        # 2. Price movement filter
        future_max = prices.shift(-price_move_window).rolling(window=price_move_window, min_periods=1).max()
        future_min = prices.shift(-price_move_window).rolling(window=price_move_window, min_periods=1).min()
        price_change = (future_max - future_min) / prices
        move_filter = price_change >= price_move_threshold

        # Combine filters
        filtered = sig_bool & persistence & move_filter

        # Preserve original dtype (numeric) for backtest compatibility
        return filtered.astype(int)

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
        *,
        entry_persist_window: int = 2,
        entry_price_move_window: int = 5,
        entry_price_move_threshold: float = 0.001,
    ) -> ComparisonResult:
        """
        Execute a side‑by‑side back‑test of manual and ML‑enhanced signals,
        applying entry confirmation filters to improve signal quality.
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

        # Apply confirmation filters to tighten entries
        filtered_manual = self._apply_entry_filters(
            manual_signals,
            prices,
            persist_window=entry_persist_window,
            price_move_window=entry_price_move_window,
            price_move_threshold=entry_price_move_threshold,
        )
        filtered_ml = self._apply_entry_filters(
            ml_signals,
            prices,
            persist_window=entry_persist_window,
            price_move_window=entry_price_move_window,
            price_move_threshold=entry_price_move_threshold,
        )

        # Ensure at least one signal remains after filtering
        if filtered_manual.sum() == 0:
            logger.warning("All manual signals filtered out; falling back to original signals.")
            filtered_manual = manual_signals
        if filtered_ml.sum() == 0:
            logger.warning("All ML signals filtered out; falling back to original signals.")
            filtered_ml = ml_signals

        # Run backtests
        manual_metrics = run_backtest(filtered_manual, prices, initial_equity)
        ml_metrics = run_backtest(filtered_ml, prices, initial_equity)

        # Benchmark data
        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Equity curve processing
        manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()

        # Statistical significance test
        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len], equal_var=False)
        else:
            t_stat, p_val = 0.0, 1.0

        # Determine winner
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
            filtered_manual_signals=int(filtered_manual.sum()),
            filtered_ml_signals=int(filtered_ml.sum()),
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