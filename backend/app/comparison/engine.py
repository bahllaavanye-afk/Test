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


class ComparisonError(Exception):
    """Custom exception for errors occurring during strategy comparison."""


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
        try:
            manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        except (ValueError, RuntimeError) as exc:
            logger.error(
                "Failed to run backtest for manual signals",
                strategy=strategy_name,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ComparisonError("Manual backtest failed") from exc

        try:
            ml_metrics = run_backtest(ml_signals, prices, initial_equity)
        except (ValueError, RuntimeError) as exc:
            logger.error(
                "Failed to run backtest for ML-enhanced signals",
                strategy=strategy_name,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ComparisonError("ML backtest failed") from exc

        try:
            benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.error(
                "Failed to fetch benchmark curves",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ComparisonError("Benchmark data retrieval failed") from exc
        except Exception as exc:  # Catch unexpected errors from async call
            logger.exception(
                "Unexpected error while fetching benchmark curves",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ComparisonError("Benchmark data retrieval failed") from exc

        try:
            benchmark_stats = get_benchmark_stats()
        except Exception as exc:
            logger.error(
                "Failed to compute benchmark statistics",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ComparisonError("Benchmark statistics computation failed") from exc

        try:
            manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
            ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
            manual_ret = manual_eq.pct_change().dropna()
            ml_ret = ml_eq.pct_change().dropna()
        except (KeyError, TypeError, AttributeError) as exc:
            logger.error(
                "Error processing equity curves for t-test",
                strategy=strategy_name,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ComparisonError("Equity curve processing failed") from exc

        try:
            min_len = min(len(manual_ret), len(ml_ret))
            if min_len > 10:
                t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
            else:
                t_stat, p_val = 0.0, 1.0
        except Exception as exc:
            logger.error(
                "Statistical test computation failed",
                strategy=strategy_name,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ComparisonError("Statistical test failed") from exc

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