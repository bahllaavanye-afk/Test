"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict

import pandas as pd
from pydantic import BaseModel, Field, validator
from scipy import stats

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


class ComparisonResult(BaseModel):
    """
    Result container for a strategy comparison run.
    """

    strategy_name: str = Field(
        ...,
        description="Human‑readable name of the strategy being evaluated.",
        example="mean_rev_20_2",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the asset under test.",
        example="AAPL",
    )
    interval: str = Field(
        ...,
        description="Timeframe of the price data (e.g., '15m', '1h').",
        example="15m",
    )
    start_date: date = Field(
        ...,
        description="Inclusive start date for the backtest period.",
        example="2023-01-01",
    )
    end_date: date = Field(
        ...,
        description="Inclusive end date for the backtest period.",
        example="2023-03-31",
    )
    manual: BacktestMetrics | None = Field(
        default=None,
        description="Backtest metrics for the manual signal set.",
    )
    ml_enhanced: BacktestMetrics | None = Field(
        default=None,
        description="Backtest metrics for the ML‑enhanced signal set.",
    )
    benchmark_curves: Dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark equity curves fetched for the period.",
    )
    benchmark_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistical summary of the benchmark curves.",
    )
    ml_improvement_sharpe: float = Field(
        default=0.0,
        description="Difference in Sharpe ratio (ML – manual).",
        example=0.12,
    )
    t_statistic: float = Field(
        default=0.0,
        description="t‑statistic from the two‑sample t‑test on returns.",
        example=1.85,
    )
    p_value: float = Field(
        default=1.0,
        description="Two‑tailed p‑value from the t‑test.",
        example=0.067,
    )
    is_significant: bool = Field(
        default=False,
        description="Whether the p‑value indicates statistical significance (< 0.05).",
    )
    winner: str = Field(
        default="neither",
        description="Identifier of the winning approach: 'ml', 'manual', or 'neither'.",
        example="ml",
    )

    @validator("end_date")
    def check_dates(cls, v: date, values: dict) -> date:
        """Ensure end_date is not earlier than start_date."""
        start = values.get("start_date")
        if start and v < start:
            raise ValueError("end_date cannot be earlier than start_date.")
        return v

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {date: lambda d: d.isoformat()}


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
        # Input validation
        if not isinstance(manual_signals, pd.Series):
            raise ValueError("manual_signals must be a pandas Series.")
        if not isinstance(ml_signals, pd.Series):
            raise ValueError("ml_signals must be a pandas Series.")
        if not isinstance(prices, pd.Series):
            raise ValueError("prices must be a pandas Series.")

        if manual_signals.empty:
            raise ValueError("manual_signals series cannot be empty.")
        if ml_signals.empty:
            raise ValueError("ml_signals series cannot be empty.")
        if prices.empty:
            raise ValueError("prices series cannot be empty.")

        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError("strategy_name must be a non‑empty string.")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non‑empty string.")
        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("interval must be a non‑empty string.")

        if not isinstance(start_date, date):
            raise ValueError("start_date must be a datetime.date instance.")
        if not isinstance(end_date, date):
            raise ValueError("end_date must be a datetime.date instance.")
        if start_date > end_date:
            raise ValueError("start_date cannot be later than end_date.")

        if not isinstance(initial_equity, (int, float)):
            raise ValueError("initial_equity must be a numeric type.")
        if initial_equity <= 0:
            raise ValueError("initial_equity must be a positive number.")

        # Align series on a common index
        common_index = manual_signals.index.intersection(ml_signals.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError(
                "manual_signals, ml_signals, and prices must share at least one common index."
            )
        manual_signals = manual_signals.loc[common_index]
        ml_signals = ml_signals.loc[common_index]
        prices = prices.loc[common_index]

        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

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
            is_significant=p_val < 0.05,
            winner=winner,
        )