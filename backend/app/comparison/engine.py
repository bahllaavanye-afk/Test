"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

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
    strategy_name: str
        Human‑readable name of the strategy being compared.
    symbol: str
        Ticker symbol the strategy was applied to.
    interval: str
        Bar interval (e.g., "5m", "1h").
    start_date: date
        Inclusive start of the back‑test period.
    end_date: date
        Inclusive end of the back‑test period.
    manual: Optional[BacktestMetrics]
        Metrics from the manual (baseline) signal set.
    ml_enhanced: Optional[BacktestMetrics]
        Metrics from the ML‑enhanced signal set.
    benchmark_curves: dict
        Benchmark equity curves fetched for the same period.
    benchmark_stats: dict
        Summary statistics of the benchmarks.
    ml_improvement_sharpe: float
        Sharpe ratio improvement of the ML version over the manual version.
    t_statistic: float
        t‑statistic from the two‑sample t‑test on returns.
    p_value: float
        p‑value from the t‑test.
    is_significant: bool
        Whether the p‑value indicates statistical significance (α = 0.05).
    winner: Literal["ml", "manual", "neither"]
        Declared winner based on Sharpe improvement.
    """

    strategy_name: str = Field(..., description="Name of the strategy under test.", example="mean_rev_20_2")
    symbol: str = Field(..., description="Ticker symbol.", example="AAPL")
    interval: str = Field(..., description="Bar interval, e.g., '15m' or '1h'.", example="15m")
    start_date: date = Field(..., description="Start date of the back‑test period.", example="2023-01-01")
    end_date: date = Field(..., description="End date of the back‑test period.", example="2023-06-30")
    manual: Optional[BacktestMetrics] = Field(
        None,
        description="Back‑test metrics for the manual signal set.",
        example=None,
    )
    ml_enhanced: Optional[BacktestMetrics] = Field(
        None,
        description="Back‑test metrics for the ML‑enhanced signal set.",
        example=None,
    )
    benchmark_curves: dict = Field(
        default_factory=dict,
        description="Dictionary of benchmark equity curves keyed by benchmark name.",
        example={"SPY": [100000, 101200, 102500]},
    )
    benchmark_stats: dict = Field(
        default_factory=dict,
        description="Statistical summary of the benchmarks.",
        example={"SPY": {"sharpe": 0.75, "max_drawdown": -0.12}},
    )
    ml_improvement_sharpe: float = Field(
        0.0,
        description="Sharpe ratio improvement of the ML version over the manual version.",
        example=0.12,
    )
    t_statistic: float = Field(
        0.0,
        description="t‑statistic from the two‑sample t‑test on returns.",
        example=1.87,
    )
    p_value: float = Field(
        1.0,
        description="p‑value from the t‑test.",
        example=0.0723,
    )
    is_significant: bool = Field(
        False,
        description="True if p‑value < 0.05, indicating statistical significance.",
        example=False,
    )
    winner: Literal["ml", "manual", "neither"] = Field(
        "neither",
        description="Declared winner based on Sharpe improvement.",
        example="ml",
    )

    @validator("strategy_name", "symbol", "interval")
    def non_empty_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non‑empty string")
        return v

    @validator("ml_improvement_sharpe", "t_statistic", "p_value")
    def numeric_values(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("must be a numeric type")
        return float(v)

    @root_validator
    def check_dates_and_winner(cls, values):
        start = values.get("start_date")
        end = values.get("end_date")
        if start and end and start > end:
            raise ValueError("start_date cannot be later than end_date")
        winner = values.get("winner")
        if winner not in {"ml", "manual", "neither"}:
            raise ValueError("winner must be one of 'ml', 'manual', or 'neither'")
        return values

    class Config:
        arbitrary_types_allowed = True
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

        # Align series on common index
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