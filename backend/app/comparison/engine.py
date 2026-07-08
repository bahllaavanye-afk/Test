"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field, validator

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


class ComparisonResult(BaseModel):
    """Result of a strategy comparison between manual and ML‑enhanced runs."""

    strategy_name: str = Field(
        ...,
        description="Name of the strategy under comparison.",
        example="mean_rev_30_2",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol for the asset being backtested.",
        example="AAPL",
    )
    interval: str = Field(
        ...,
        description="Timeframe of the price data (e.g., '1h', 'daily').",
        example="1h",
    )
    start_date: date = Field(
        ...,
        description="Inclusive start date of the backtest period.",
        example="2023-01-01",
    )
    end_date: date = Field(
        ...,
        description="Inclusive end date of the backtest period.",
        example="2023-06-30",
    )
    manual: BacktestMetrics | None = Field(
        None,
        description="Metrics from the manual‑signal backtest.",
    )
    ml_enhanced: BacktestMetrics | None = Field(
        None,
        description="Metrics from the ML‑enhanced signal backtest.",
    )
    benchmark_curves: dict = Field(
        default_factory=dict,
        description="Dictionary of benchmark equity curves keyed by benchmark name.",
        example={"SP500": [{"date": "2023-01-01", "equity": 100000}, {"date": "2023-01-02", "equity": 100500}]},
    )
    benchmark_stats: dict = Field(
        default_factory=dict,
        description="Statistical summary of benchmarks (e.g., Sharpe, MaxDD).",
        example={"SP500": {"sharpe": 1.2, "max_dd": 0.08}},
    )
    ml_improvement_sharpe: float = Field(
        0.0,
        description="Improvement in Sharpe ratio (ML – manual).",
        example=0.15,
    )
    t_statistic: float = Field(
        0.0,
        description="t‑statistic from the two‑sample t‑test.",
        example=2.34,
    )
    p_value: float = Field(
        1.0,
        description="p‑value from the significance test.",
        example=0.0123,
    )
    is_significant: bool = Field(
        False,
        description="Whether the result is statistically significant (p < 0.05).",
    )
    winner: str = Field(
        "neither",
        description="Identified winner: 'ml', 'manual', or 'neither'.",
        example="ml",
    )

    @validator("end_date")
    def validate_date_order(cls, v, values):
        start = values.get("start_date")
        if start and v < start:
            raise ValueError("end_date must be on or after start_date")
        return v

    @validator("p_value")
    def validate_p_value_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("p_value must be between 0 and 1")
        return v

    @validator("winner")
    def validate_winner(cls, v):
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