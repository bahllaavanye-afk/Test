"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, root_validator, validator
from scipy import stats

from app.backtest.engine import BacktestMetrics, run_backtest
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


class ComparisonResult(BaseModel):
    """Result of a strategy comparison run.

    Attributes
    ----------
    strategy_name: Name of the strategy under test.
    symbol: Trading symbol (e.g., 'AAPL').
    interval: Data interval (e.g., '1d', '5m').
    start_date: Start of the backtest period.
    end_date: End of the backtest period.
    manual: Metrics from the manual signal backtest.
    ml_enhanced: Metrics from the ML‑enhanced signal backtest.
    benchmark_curves: Raw benchmark equity curves.
    benchmark_stats: Summary statistics for the benchmarks.
    ml_improvement_sharpe: Sharpe improvement of ML over manual.
    t_statistic: t‑statistic from the two‑sample t‑test.
    p_value: P‑value from the t‑test.
    is_significant: Whether the p‑value indicates statistical significance (<0.05).
    winner: Which approach performed better ('ml', 'manual', or 'neither').
    """

    strategy_name: str = Field(..., description="Name of the strategy being compared.", example="mean_rev_20_1.5")
    symbol: str = Field(..., description="Ticker symbol for the asset.", example="AAPL")
    interval: str = Field(..., description="Data frequency interval.", example="1d")
    start_date: date = Field(..., description="Start date of the backtest period.", example="2023-01-01")
    end_date: date = Field(..., description="End date of the backtest period.", example="2023-12-31")
    manual: Optional[BacktestMetrics] = Field(
        default=None,
        description="Performance metrics from the manual signal backtest.",
    )
    ml_enhanced: Optional[BacktestMetrics] = Field(
        default=None,
        description="Performance metrics from the ML‑enhanced signal backtest.",
    )
    benchmark_curves: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of benchmark equity curves keyed by benchmark name.",
    )
    benchmark_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregated statistics for the benchmark curves.",
    )
    ml_improvement_sharpe: float = Field(
        ...,
        description="Difference in Sharpe ratio (ML – manual).",
        example=0.12,
    )
    t_statistic: float = Field(
        ...,
        description="t‑statistic from the two‑sample t‑test comparing returns.",
        example=1.85,
    )
    p_value: float = Field(
        ...,
        description="P‑value from the t‑test; low values indicate significance.",
        example=0.067,
    )
    is_significant: bool = Field(
        ...,
        description="True if p_value < 0.05, indicating statistical significance.",
        example=False,
    )
    winner: str = Field(
        ...,
        description="Identifier of the better performing approach.",
        example="ml",
    )

    @root_validator
    def check_dates(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure start_date is not after end_date."""
        start = values.get("start_date")
        end = values.get("end_date")
        if start and end and start > end:
            raise ValueError("start_date must be on or before end_date")
        return values

    @validator("winner")
    def validate_winner(cls, v: str) -> str:
        """Validate that winner is one of the allowed options."""
        allowed = {"ml", "manual", "neither"}
        if v not in allowed:
            raise ValueError(f"winner must be one of {allowed}")
        return v


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
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Extract daily return series for t-test
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