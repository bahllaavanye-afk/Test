"""
Strategy Comparison Engine.

Runs a manual strategy and its ML‑enhanced counterpart over the same period,
compares their performance against benchmark curves, and computes statistical
significance of any improvement.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from app.backtest.engine import BacktestMetrics, run_backtest
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


@dataclass
class ComparisonResult:
    """
    Container for the outcome of a strategy comparison.

    Attributes
    ----------
    strategy_name: str
        Human‑readable name of the strategy being evaluated.
    symbol: str
        Ticker symbol the strategy was applied to.
    interval: str
        Data interval (e.g., "1h", "daily").
    start_date: date
        Inclusive start date of the backtest period.
    end_date: date
        Inclusive end date of the backtest period.
    manual: BacktestMetrics | None
        Metrics from the manual (non‑ML) version of the strategy.
    ml_enhanced: BacktestMetrics | None
        Metrics from the ML‑enhanced version of the strategy.
    benchmark_curves: Dict[str, Any]
        Time‑series benchmark performance curves fetched for the period.
    benchmark_stats: Dict[str, Any]
        Summary statistics derived from the benchmark curves.
    ml_improvement_sharpe: float
        Sharpe ratio improvement of the ML version over the manual version.
    t_statistic: float
        t‑statistic from the two‑sample t‑test on daily returns.
    p_value: float
        Two‑tailed p‑value from the t‑test.
    is_significant: bool
        Whether the p‑value indicates statistical significance (p < 0.05).
    winner: str
        Identifier of the better performing version ("ml", "manual", or "neither").
    """

    strategy_name: str
    symbol: str
    interval: str
    start_date: date
    end_date: date
    manual: BacktestMetrics | None = None
    ml_enhanced: BacktestMetrics | None = None
    benchmark_curves: Dict[str, Any] = field(default_factory=dict)
    benchmark_stats: Dict[str, Any] = field(default_factory=dict)
    ml_improvement_sharpe: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    winner: str = "neither"


class StrategyComparisonEngine:
    """Engine responsible for executing and analysing strategy comparisons."""

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
        """
        Execute backtests for manual and ML‑enhanced signal sets, fetch benchmarks,
        and compute comparative statistics.

        Parameters
        ----------
        manual_signals : pd.Series
            Signal series for the manual version of the strategy.
        ml_signals : pd.Series
            Signal series for the ML‑enhanced version of the strategy.
        prices : pd.Series
            Corresponding price series used for both backtests.
        strategy_name : str
            Name of the strategy under comparison.
        symbol : str
            Ticker symbol the strategy trades.
        interval : str
            Data granularity (e.g., "1h", "daily").
        start_date : date
            Start date of the backtest period (inclusive).
        end_date : date
            End date of the backtest period (inclusive).
        initial_equity : float, optional
            Starting capital for the backtest; defaults to 100,000.

        Returns
        -------
        ComparisonResult
            Populated dataclass instance containing backtest metrics,
            benchmark information, and statistical test outcomes.
        """
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Extract daily equity curves and compute returns for the t‑test
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