"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Tuple

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
    def _validate_signals(
        self,
        signals: pd.Series,
        prices: pd.Series,
        name: str,
        confirmation: int,
    ) -> pd.Series:
        """
        Validate and tighten signal quality.

        - Ensure no NaNs; replace with 0 (no signal).
        - Force binary signals (0/1); any non‑zero value becomes 1.
        - Align index with price series; raise if mis‑aligned.
        - Apply a confirmation filter: a signal is only kept if it
          persists unchanged for `confirmation` consecutive periods.
        """
        if not isinstance(signals, pd.Series):
            raise TypeError(f"{name} signals must be a pandas Series")

        # Align indices
        if not signals.index.equals(prices.index):
            # Reindex to price index, forward fill then fill remaining NaNs with 0
            signals = signals.reindex(prices.index).fillna(method="ffill").fillna(0)

        # Remove NaNs and enforce binary
        signals = signals.fillna(0).astype(int).clip(lower=0, upper=1)

        # Confirmation filter
        if confirmation > 1:
            # Rolling window must have identical values; otherwise set to 0
            confirmed = (
                signals.rolling(window=confirmation, min_periods=confirmation)
                .apply(lambda x: x.iloc[-1] if len(set(x)) == 1 else 0, raw=False)
            )
            # First `confirmation-1` entries become 0 because window not full
            confirmed = confirmed.fillna(0).astype(int)
            signals = confirmed

        return signals

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
        signal_confirmation: int = 2,
    ) -> ComparisonResult:
        """
        Run a side‑by‑side backtest of manual and ML‑enhanced signals,
        then compare performance against benchmarks.

        The `signal_confirmation` parameter adds a confirmation filter
        to both signal sets, tightening entry conditions and reducing
        spurious trades.
        """
        # Validate and filter signals
        manual_sig = self._validate_signals(
            manual_signals, prices, "manual", signal_confirmation
        )
        ml_sig = self._validate_signals(
            ml_signals, prices, "ml", signal_confirmation
        )

        # Run backtests
        manual_metrics = run_backtest(manual_sig, prices, initial_equity)
        ml_metrics = run_backtest(ml_sig, prices, initial_equity)

        # Fetch benchmarks
        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Prepare equity return series for statistical testing
        manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()

        # Ensure comparable lengths
        min_len = min(len(manual_ret), len(ml_ret))
        if min_len < 5:
            t_stat, p_val = 0.0, 1.0
        else:
            # Use Welch's t‑test (unequal variance) for robustness
            t_stat, p_val = stats.ttest_ind(
                ml_ret.iloc[:min_len],
                manual_ret.iloc[:min_len],
                equal_var=False,
            )

        # Sharpe improvement and winner determination
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
            signal_confirmation=signal_confirmation,
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