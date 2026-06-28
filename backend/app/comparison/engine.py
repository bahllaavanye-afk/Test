"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""

from __future__ import annotations

from dataclasses import field
from datetime import date
from typing import Dict, Tuple, Awaitable

import pandas as pd
from pydantic import BaseModel, Field, validator
from scipy import stats

from app.backtest.engine import BacktestMetrics, run_backtest
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger

# ------------------------------
# Constants
# ------------------------------
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

EQUITY_KEY: str = "equity"
DEFAULT_T_STAT: float = 0.0
DEFAULT_P_VAL: float = 1.0


class ComparisonResult(BaseModel):
    """Result of a strategy comparison between manual and ML‑enhanced signals."""

    strategy_name: str = Field(
        ...,
        description="Human‑readable identifier for the strategy under test.",
        example="rsi_10_25_75",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol the strategy was applied to.",
        example="AAPL",
    )
    interval: str = Field(
        ...,
        description="Timeframe of the price data (e.g., '1h', 'daily').",
        example="1h",
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
        default=None,
        description="Metrics from the backtest using manual signals.",
    )
    ml_enhanced: BacktestMetrics | None = Field(
        default=None,
        description="Metrics from the backtest using ML‑enhanced signals.",
    )
    benchmark_curves: dict = Field(
        default_factory=dict,
        description="Benchmark equity curves over the same period.",
        example={"SP500": [100000, 101200, 102500]},
    )
    benchmark_stats: dict = Field(
        default_factory=dict,
        description="Statistical summary of benchmark performance.",
        example={"SP500_sharpe": 0.85},
    )
    ml_improvement_sharpe: float = Field(
        default=0.0,
        description="Absolute Sharpe improvement of the ML strategy over manual.",
        example=0.15,
    )
    t_statistic: float = Field(
        default=0.0,
        description="t‑statistic from the two‑sample test of returns.",
        example=1.23,
    )
    p_value: float = Field(
        default=1.0,
        description="p‑value associated with the t‑statistic.",
        example=0.215,
    )
    is_significant: bool = Field(
        default=False,
        description="Whether the p‑value is below the significance threshold.",
        example=False,
    )
    winner: str = Field(
        default=WINNER_NEITHER,
        description="Identifier of the winning approach ('ml', 'manual', or 'neither').",
        example="ml",
    )

    @validator("end_date")
    def check_dates(cls, v: date, values: dict) -> date:
        """Ensure end_date is not earlier than start_date."""
        start = values.get("start_date")
        if start and v < start:
            raise ValueError("end_date must be on or after start_date")
        return v

    @validator("winner")
    def validate_winner(cls, v: str) -> str:
        """Validate that winner is one of the predefined constants."""
        allowed = {WINNER_ML, WINNER_MANUAL, WINNER_NEITHER}
        if v not in allowed:
            raise ValueError(f"winner must be one of {allowed}")
        return v

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {date: lambda d: d.isoformat()}


class StrategyComparisonEngine:
    """Engine that compares manual and ML‑enhanced trading strategies."""

    # Simple in‑memory cache for benchmark curves keyed by (start_date, end_date)
    _benchmark_cache: Dict[Tuple[date, date], dict] = {}

    async def _get_benchmark_curves(self, start: date, end: date) -> dict:
        """
        Retrieve benchmark equity curves for the given period.

        Uses an in‑memory cache to avoid repeated asynchronous fetches.

        Args:
            start: Inclusive start date for the benchmark data.
            end: Inclusive end date for the benchmark data.

        Returns:
            A dictionary containing benchmark curves.
        """
        cache_key = (start, end)
        if cache_key not in self._benchmark_cache:
            self._benchmark_cache[cache_key] = await fetch_benchmark_curves(start, end)
        return self._benchmark_cache[cache_key]

    def _run_backtests(
        self,
        manual_signals: pd.Series,
        ml_signals: pd.Series,
        prices: pd.Series,
        initial_equity: float,
    ) -> Tuple[BacktestMetrics, BacktestMetrics]:
        """
        Execute backtests for manual and ML‑enhanced signals.

        Args:
            manual_signals: Series of binary/manual signals.
            ml_signals: Series of binary/ML‑enhanced signals.
            prices: Series of price data aligned with the signals.
            initial_equity: Starting capital for the backtest.

        Returns:
            A tuple containing metrics for the manual and ML backtests.
        """
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)
        return manual_metrics, ml_metrics

    def _extract_equity_series(self, metrics: BacktestMetrics) -> pd.Series:
        """
        Convert a BacktestMetrics equity_curve into a pandas Series of equity values.

        Args:
            metrics: BacktestMetrics object containing an equity curve.

        Returns:
            pandas Series of equity values.
        """
        return pd.Series([e[EQUITY_KEY] for e in metrics.equity_curve])

    def _compute_statistics(
        self,
        manual_eq: pd.Series,
        ml_eq: pd.Series,
    ) -> Tuple[float, float]:
        """
        Calculate t‑statistic and p‑value for the equity return series.

        Args:
            manual_eq: Equity series from the manual backtest.
            ml_eq: Equity series from the ML‑enhanced backtest.

        Returns:
            Tuple of (t_statistic, p_value). If data length is insufficient,
            returns default constants.
        """
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()
        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > MIN_DATA_LENGTH:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
        else:
            t_stat, p_val = DEFAULT_T_STAT, DEFAULT_P_VAL
        return t_stat, p_val

    def _determine_winner(
        self,
        ml_sharpe: float,
        manual_sharpe: float,
    ) -> Tuple[str, float]:
        """
        Decide the winner based on Sharpe improvement and a predefined threshold.

        Args:
            ml_sharpe: Sharpe ratio of the ML‑enhanced strategy.
            manual_sharpe: Sharpe ratio of the manual strategy.

        Returns:
            Tuple of (winner identifier, Sharpe improvement).
        """
        improvement = ml_sharpe - manual_sharpe
        if abs(improvement) < IMPROVEMENT_THRESHOLD:
            winner = WINNER_NEITHER
        else:
            winner = WINNER_ML if ml_sharpe > manual_sharpe else WINNER_MANUAL
        return winner, improvement

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
        """
        Run the full comparison pipeline and produce a structured result.

        This includes backtesting both signal sets, fetching benchmark data,
        statistical testing, and winner determination.

        Args:
            manual_signals: Series of manual trading signals.
            ml_signals: Series of ML‑enhanced trading signals.
            prices: Series of price data aligned with the signals.
            strategy_name: Human‑readable name of the strategy.
            symbol: Ticker symbol.
            interval: Data frequency (e.g., '1h', 'daily').
            start_date: Inclusive start date for the backtest period.
            end_date: Inclusive end date for the backtest period.
            initial_equity: Starting capital; defaults to DEFAULT_INITIAL_EQUITY.

        Returns:
            A ComparisonResult instance containing all relevant metrics.
        """
        # Run backtests (potentially expensive)
        manual_metrics, ml_metrics = self._run_backtests(
            manual_signals, ml_signals, prices, initial_equity
        )

        # Cached retrieval of benchmark data
        benchmark_curves = await self._get_benchmark_curves(start_date, end_date)
        benchmark_stats = get_benchmark_stats()

        # Extract equity series for statistical analysis
        manual_eq = self._extract_equity_series(manual_metrics)
        ml_eq = self._extract_equity_series(ml_metrics)

        # Compute t‑statistic and p‑value
        t_stat, p_val = self._compute_statistics(manual_eq, ml_eq)

        # Determine winner and Sharpe improvement
        winner, improvement = self._determine_winner(ml_metrics.sharpe, manual_metrics.sharpe)

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
            t_statistic=round(t_stat, TSTAT_ROUND),
            p_value=round(p_val, PVAL_ROUND),
            is_significant=p_val < SIGNIFICANCE_LEVEL,
            winner=winner,
        )