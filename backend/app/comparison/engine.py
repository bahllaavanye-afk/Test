"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Tuple

import pandas as pd
from scipy import stats

from app.backtest.engine import BacktestMetrics, run_backtest
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger

# Constants
DEFAULT_INITIAL_EQUITY: float = 100_000
MIN_DATA_LENGTH: int = 10
IMPROVEMENT_THRESHOLD: float = 0.1
SIGNIFICANCE_LEVEL: float = 0.05
LOG_PVALUE_PRECISION: int = 4
IMPROVEMENT_ROUND: int = 4
TSTAT_ROUND: int = 4
PVAL_ROUND: int = 6
WINNER_ML: str = "ml"
WINNER_MANUAL: str = "manual"
WINNER_NEITHER: str = "neither"
LOG_MESSAGE: str = "Comparison complete"

# Extracted magic strings and default numeric values
EQUITY_KEY: str = "equity"
DEFAULT_T_STAT: float = 0.0
DEFAULT_P_VAL: float = 1.0


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
    # Simple in‑memory cache for benchmark curves keyed by (start_date, end_date)
    _benchmark_cache: Dict[Tuple[date, date], dict] = {}

    async def _get_benchmark_curves(self, start: date, end: date) -> dict:
        """Return benchmark curves, using an in‑memory cache to avoid repeated async fetches."""
        cache_key = (start, end)
        if cache_key not in self._benchmark_cache:
            self._benchmark_cache[cache_key] = await fetch_benchmark_curves(start, end)
        return self._benchmark_cache[cache_key]

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
        # Run backtests (potentially expensive)
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        # Cached retrieval of benchmark data
        benchmark_curves = await self._get_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Vectorized extraction of equity curves
        manual_eq = pd.Series([e[EQUITY_KEY] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e[EQUITY_KEY] for e in ml_metrics.equity_curve])

        # Compute daily returns
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()

        # Early‑exit for insufficient data
        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > MIN_DATA_LENGTH:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
        else:
            t_stat, p_val = DEFAULT_T_STAT, DEFAULT_P_VAL

        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = WINNER_ML if ml_metrics.sharpe > manual_metrics.sharpe else WINNER_MANUAL
        if abs(improvement) < IMPROVEMENT_THRESHOLD:
            winner = WINNER_NEITHER

        logger.info(
            LOG_MESSAGE,
            strategy=strategy_name,
            manual_sharpe=manual_metrics.sharpe,
            ml_sharpe=ml_metrics.sharpe,
            p_value=round(p_val, LOG_PVALUE_PRECISION),
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
            t_statistic=round(float(t_stat), TSTAT_ROUND),
            p_value=round(float(p_val), PVAL_ROUND),
            is_significant=(p_val < SIGNIFICANCE_LEVEL),
            winner=winner,
        )