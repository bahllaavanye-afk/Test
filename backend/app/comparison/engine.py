"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import date
import traceback

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
        # Initialise placeholders
        manual_metrics: BacktestMetrics | None = None
        ml_metrics: BacktestMetrics | None = None
        benchmark_curves: dict = {}
        benchmark_stats: dict = {}

        # Run manual backtest
        try:
            manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        except ValueError as ve:
            logger.error(
                "ValueError during manual backtest",
                strategy=strategy_name,
                error=str(ve),
                exc_info=True,
            )
        except RuntimeError as re:
            logger.error(
                "RuntimeError during manual backtest",
                strategy=strategy_name,
                error=str(re),
                exc_info=True,
            )
        except Exception as e:
            logger.exception(
                "Unexpected error during manual backtest",
                strategy=strategy_name,
                error=str(e),
            )

        # Run ML‑enhanced backtest
        try:
            ml_metrics = run_backtest(ml_signals, prices, initial_equity)
        except ValueError as ve:
            logger.error(
                "ValueError during ML backtest",
                strategy=strategy_name,
                error=str(ve),
                exc_info=True,
            )
        except RuntimeError as re:
            logger.error(
                "RuntimeError during ML backtest",
                strategy=strategy_name,
                error=str(re),
                exc_info=True,
            )
        except Exception as e:
            logger.exception(
                "Unexpected error during ML backtest",
                strategy=strategy_name,
                error=str(e),
            )

        # Fetch benchmarks
        try:
            benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        except asyncio.TimeoutError as te:
            logger.error(
                "Timeout while fetching benchmark curves",
                start_date=str(start_date),
                end_date=str(end_date),
                error=str(te),
                exc_info=True,
            )
        except Exception as e:
            logger.exception(
                "Failed to fetch benchmark curves",
                start_date=str(start_date),
                end_date=str(end_date),
                error=str(e),
            )

        # Get benchmark statistics
        try:
            benchmark_stats = get_benchmark_stats()
        except Exception as e:
            logger.exception(
                "Failed to retrieve benchmark statistics",
                error=str(e),
            )

        # If either backtest failed, skip statistical comparison
        if manual_metrics is None or ml_metrics is None:
            logger.warning(
                "One or both backtests failed; returning partial result",
                strategy=strategy_name,
                manual_success=manual_metrics is not None,
                ml_success=ml_metrics is not None,
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
                ml_improvement_sharpe=0.0,
                t_statistic=0.0,
                p_value=1.0,
                is_significant=False,
                winner="neither",
            )

        # Extract daily return series for t‑test
        try:
            manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
            ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
            manual_ret = manual_eq.pct_change().dropna()
            ml_ret = ml_eq.pct_change().dropna()
        except (KeyError, TypeError) as e:
            logger.exception(
                "Error extracting equity curves for t‑test",
                strategy=strategy_name,
                error=str(e),
            )
            manual_ret = pd.Series(dtype=float)
            ml_ret = pd.Series(dtype=float)

        # Perform statistical test
        try:
            min_len = min(len(manual_ret), len(ml_ret))
            if min_len > 10:
                t_stat, p_val = stats.ttest_ind(
                    ml_ret.iloc[:min_len], manual_ret.iloc[:min_len]
                )
            else:
                t_stat, p_val = 0.0, 1.0
        except Exception as e:
            logger.exception(
                "Statistical test failed",
                strategy=strategy_name,
                error=str(e),
            )
            t_stat, p_val = 0.0, 1.0

        # Determine improvement and winner
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