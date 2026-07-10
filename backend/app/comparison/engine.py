"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from scipy import stats

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


class ComparisonResult(BaseModel):
    """Result of a strategy comparison run."""

    strategy_name: str = Field(
        ...,
        description="Human‑readable name of the strategy under comparison.",
        example="mean_rev_20_1.5",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol the strategy was applied to.",
        example="SPY",
    )
    interval: str = Field(
        ...,
        description="Data granularity (e.g., '1d', '5m').",
        example="1d",
    )
    start_date: date = Field(
        ...,
        description="Inclusive start date for the backtest period.",
        example="2023-01-01",
    )
    end_date: date = Field(
        ...,
        description="Inclusive end date for the backtest period.",
        example="2023-12-31",
    )
    manual: BacktestMetrics | None = Field(
        None,
        description="Backtest metrics for the manual signal set.",
    )
    ml_enhanced: BacktestMetrics | None = Field(
        None,
        description="Backtest metrics for the ML‑enhanced signal set.",
    )
    benchmark_curves: Dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark equity curves fetched for the period.",
    )
    benchmark_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistical summary of benchmark performance.",
    )
    ml_improvement_sharpe: float = Field(
        0.0,
        description="Difference in Sharpe ratio (ML – manual).",
        example=0.12,
    )
    t_statistic: float = Field(
        0.0,
        description="t‑statistic from the two‑sample t‑test.",
        example=1.85,
    )
    p_value: float = Field(
        1.0,
        description="p‑value from the t‑test indicating statistical significance.",
        example=0.067,
    )
    is_significant: bool = Field(
        False,
        description="True if p_value < 0.05, indicating a statistically significant difference.",
    )
    winner: str = Field(
        "neither",
        description="Identifier of the better performing strategy: 'ml', 'manual', or 'neither'.",
        example="ml",
    )

    @validator("end_date")
    def check_date_order(cls, v: date, values: dict) -> date:
        start = values.get("start_date")
        if start and v < start:
            raise ValueError("end_date must be on or after start_date")
        return v

    @validator("winner")
    def validate_winner(cls, v: str) -> str:
        if v not in {"ml", "manual", "neither"}:
            raise ValueError("winner must be one of 'ml', 'manual', or 'neither'")
        return v

    @validator("p_value")
    def p_value_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("p_value must be between 0 and 1")
        return v

    class Config:
        orm_mode = True


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
        """Run a side‑by‑side backtest of manual vs ML‑enhanced signals."""
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")

        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Extract daily return series for t‑test
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