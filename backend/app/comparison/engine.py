"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

from dataclasses import field
from datetime import date
from typing import Dict, Optional

import pandas as pd
from pydantic import BaseModel, Field, root_validator, validator
from scipy import stats

from app.backtest.engine import BacktestMetrics, run_backtest
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


class ComparisonResult(BaseModel):
    """Result of a strategy comparison between manual and ML‑enhanced runs."""

    strategy_name: str = Field(
        ...,
        description="Human readable name of the strategy under comparison.",
        example="MeanReversion_20_2",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the instrument the strategy was run on.",
        example="AAPL",
    )
    interval: str = Field(
        ...,
        description="Timeframe of the price data, e.g., '1d', '5m'.",
        example="1d",
    )
    start_date: date = Field(
        ...,
        description="Inclusive start date of the back‑test period.",
        example="2023-01-01",
    )
    end_date: date = Field(
        ...,
        description="Inclusive end date of the back‑test period.",
        example="2023-12-31",
    )
    manual: Optional[BacktestMetrics] = Field(
        None,
        description="Metrics generated from the manual signal set.",
    )
    ml_enhanced: Optional[BacktestMetrics] = Field(
        None,
        description="Metrics generated from the ML‑enhanced signal set.",
    )
    benchmark_curves: Dict = Field(
        default_factory=dict,
        description="Benchmark equity curves keyed by benchmark name.",
        example={"SP500": [{"date": "2023-01-01", "equity": 100000}, ...]},
    )
    benchmark_stats: Dict = Field(
        default_factory=dict,
        description="Statistical summary of benchmark performance.",
        example={"SP500": {"sharpe": 0.85, "max_drawdown": -0.12}},
    )
    ml_improvement_sharpe: float = Field(
        0.0,
        description="Difference in Sharpe ratio (ML – manual).",
        example=0.15,
    )
    t_statistic: float = Field(
        0.0,
        description="t‑statistic from the two‑sample t‑test comparing returns.",
        example=2.34,
    )
    p_value: float = Field(
        1.0,
        description="p‑value from the t‑test; lower values indicate statistical significance.",
        example=0.021,
    )
    is_significant: bool = Field(
        False,
        description="True if p_value < 0.05, indicating a statistically significant difference.",
    )
    winner: str = Field(
        "neither",
        description="Identifier of the winning approach: 'ml', 'manual', or 'neither'.",
        example="ml",
    )

    @validator("strategy_name", "symbol", "interval")
    def non_empty_str(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non‑empty string")
        return v

    @validator("ml_improvement_sharpe", "t_statistic", "p_value")
    def non_negative_float(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non‑negative")
        return v

    @root_validator
    def check_dates_and_winner(cls, values):
        start, end = values.get("start_date"), values.get("end_date")
        if start and end and start > end:
            raise ValueError("start_date cannot be later than end_date")

        winner = values.get("winner")
        if winner not in {"ml", "manual", "neither"}:
            raise ValueError("winner must be one of 'ml', 'manual', or 'neither'")
        return values


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
            raise ValueError("strategy_name must be a non-empty string.")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("interval must be a non-empty string.")

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

        # Align series on the same index (optional but helps consistency)
        common_index = manual_signals.index.intersection(ml_signals.index).intersection(prices.index)
        if common_index.empty:
            raise ValueError("manual_signals, ml_signals, and prices must share at least one common index.")
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