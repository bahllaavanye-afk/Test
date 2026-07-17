"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

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
    """Engine that compares a manual signal series with an ML‑enhanced series.

    The engine performs lightweight signal sanitisation to tighten entry
    conditions, add confirmation filters and improve exit handling before
    delegating to the back‑test runner.
    """

    _persistence: int = 2          # minimum consecutive periods for a valid entry
    _price_confirmation: float = 0.001  # 0.1% price move confirming the signal
    _max_holding: int = 20        # maximum bars to hold a position
    _stop_loss: float = 0.02     # 2% stop‑loss from entry price

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
        # Align and sanitise signals
        manual_signals = self._align_signals(manual_signals, prices)
        ml_signals = self._align_signals(ml_signals, prices)

        manual_signals = self._filter_signals(manual_signals, prices)
        ml_signals = self._filter_signals(ml_signals, prices)

        # Run backtests
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        # Benchmark data
        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Prepare equity series for statistical test
        manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])

        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()

        # Use independent t‑test only when enough observations are available
        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(
                ml_ret.iloc[:min_len], manual_ret.iloc[:min_len], equal_var=False
            )
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

    def _align_signals(self, signals: pd.Series, prices: pd.Series) -> pd.Series:
        """Align signal index with price index and drop NaNs."""
        aligned = signals.reindex(prices.index).fillna(0)
        aligned = aligned.astype(float)
        return aligned

    def _filter_signals(self, signals: pd.Series, prices: pd.Series) -> pd.Series:
        """Apply entry tightening, confirmation, and exit rules."""
        # 1. Persistence filter – keep signal only if it repeats for _persistence periods
        persisted = signals.copy()
        for i in range(1, self._persistence):
            persisted &= signals.shift(i) == signals
        persisted = persisted.astype(float)

        # 2. Price confirmation – require price to move in signal direction
        direction = np.sign(persisted)
        price_change = prices.pct_change().fillna(0)
        confirmation = (direction * price_change) >= self._price_confirmation
        confirmed = persisted * confirmation.astype(float)

        # 3. Enforce max holding period and stop‑loss
        positions = confirmed.cumsum() * direction
        entry_price = prices.where(confirmed != 0).ffill()
        holding_days = (positions != 0).groupby((positions == 0).cumsum()).cumcount() + 1

        stop_loss_trigger = (prices - entry_price) / entry_price <= -self._stop_loss
        exit_condition = (holding_days >= self._max_holding) | stop_loss_trigger

        cleaned = confirmed.copy()
        cleaned[exit_condition] = 0.0
        return cleaned.astype(float)