"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


class ComparisonEngineError(Exception):
    """Base exception for errors raised by StrategyComparisonEngine."""


class BacktestError(ComparisonEngineError):
    """Raised when backtesting fails."""


class BenchmarkError(ComparisonEngineError):
    """Raised when fetching benchmark data fails."""


class StatisticalError(ComparisonEngineError):
    """Raised when statistical calculations fail."""


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
        """
        Execute a comparison between a manual and an ML‑enhanced strategy.

        Parameters
        ----------
        manual_signals, ml_signals, prices : pd.Series
            Input time‑series data.
        strategy_name, symbol, interval : str
            Metadata describing the run.
        start_date, end_date : date
            Period for benchmark data.
        initial_equity : float, optional
            Starting equity for backtests.

        Returns
        -------
        ComparisonResult
            Aggregated comparison outcomes.

        Raises
        ------
        BacktestError
            If either backtest execution fails.
        BenchmarkError
            If benchmark data cannot be retrieved.
        StatisticalError
            If statistical calculations encounter an error.
        """
        # ------------------------------------------------------------------
        # Run manual backtest
        # ------------------------------------------------------------------
        try:
            manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        except Exception as exc:
            logger.error(
                "Manual backtest failed",
                strategy=strategy_name,
                error=str(exc),
                exc_info=True,
            )
            raise BacktestError("Manual backtest execution failed") from exc

        # ------------------------------------------------------------------
        # Run ML‑enhanced backtest
        # ------------------------------------------------------------------
        try:
            ml_metrics = run_backtest(ml_signals, prices, initial_equity)
        except Exception as exc:
            logger.error(
                "ML‑enhanced backtest failed",
                strategy=strategy_name,
                error=str(exc),
                exc_info=True,
            )
            raise BacktestError("ML‑enhanced backtest execution failed") from exc

        # ------------------------------------------------------------------
        # Fetch benchmark data
        # ------------------------------------------------------------------
        try:
            benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
            benchmark_stats = get_benchmark_stats()
        except Exception as exc:
            logger.error(
                "Benchmark retrieval failed",
                start_date=start_date,
                end_date=end_date,
                error=str(exc),
                exc_info=True,
            )
            raise BenchmarkError("Failed to fetch benchmark data") from exc

        # ------------------------------------------------------------------
        # Prepare equity series for statistical comparison
        # ------------------------------------------------------------------
        try:
            manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
            ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
            manual_ret = manual_eq.pct_change().dropna()
            ml_ret = ml_eq.pct_change().dropna()
        except (KeyError, AttributeError, TypeError) as exc:
            logger.error(
                "Equity curve extraction failed",
                manual_metrics=repr(manual_metrics),
                ml_metrics=repr(ml_metrics),
                error=str(exc),
                exc_info=True,
            )
            raise StatisticalError("Failed to extract equity curves") from exc

        # ------------------------------------------------------------------
        # Perform t‑test (if sufficient data)
        # ------------------------------------------------------------------
        try:
            min_len = min(len(manual_ret), len(ml_ret))
            if min_len > 10:
                t_stat, p_val = stats.ttest_ind(
                    ml_ret.iloc[:min_len], manual_ret.iloc[:min_len], equal_var=False
                )
            else:
                t_stat, p_val = 0.0, 1.0
        except Exception as exc:
            logger.error(
                "Statistical test failed",
                min_len=min_len,
                error=str(exc),
                exc_info=True,
            )
            raise StatisticalError("T‑test calculation failed") from exc

        # ------------------------------------------------------------------
        # Compute improvement and winner
        # ------------------------------------------------------------------
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