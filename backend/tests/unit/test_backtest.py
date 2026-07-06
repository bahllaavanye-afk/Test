"""Unit tests for backtest engine."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
from app.backtest.engine import run_backtest


def make_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate a synthetic price series.

    Args:
        n: Number of price points.
        seed: Random seed for reproducibility.

    Returns:
        A pandas Series indexed by dates.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices = 100 * np.cumprod(1 + returns)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx)


class BacktestMetricsSchema(BaseModel):
    """Schema representing the output of a backtest run."""

    sharpe: float = Field(
        ...,
        description="Annualized Sharpe ratio of the strategy.",
        example=1.25,
    )
    max_drawdown: float = Field(
        ...,
        description="Maximum drawdown expressed as a negative fraction of peak equity.",
        example=-0.18,
    )
    win_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of winning trades (0‑1).",
        example=0.57,
    )
    equity_curve: pd.Series = Field(
        ...,
        description="Equity curve over time, indexed by timestamps.",
        example=pd.Series([100, 101, 102], index=pd.date_range("2020-01-01", periods=3)),
    )
    num_trades: int = Field(
        ...,
        ge=0,
        description="Total number of executed trades.",
        example=23,
    )

    @validator("max_drawdown")
    def check_max_drawdown_negative(cls, v):
        if v > 0:
            raise ValueError("max_drawdown must be non‑positive")
        return v


def _metrics_to_dict(metrics) -> dict:
    """Convert a metrics object to a dict suitable for Pydantic validation."""
    if isinstance(metrics, dict):
        return metrics
    if hasattr(metrics, "dict"):
        return metrics.dict()
    # Fallback: extract public attributes
    return {
        key: getattr(metrics, key)
        for key in dir(metrics)
        if not key.startswith("_") and not callable(getattr(metrics, key))
    }


def test_backtest_buy_and_hold():
    """Validate that a simple buy‑and‑hold strategy produces sensible metrics."""
    prices = make_prices()
    signals = pd.Series(1, index=prices.index)
    raw_metrics = run_backtest(signals, prices)
    metrics = BacktestMetricsSchema(**_metrics_to_dict(raw_metrics))

    assert metrics.sharpe is not None
    assert -1.0 <= metrics.max_drawdown <= 0.0
    assert 0.0 <= metrics.win_rate <= 1.0
    assert len(metrics.equity_curve) > 0


def test_backtest_empty_signals():
    """Ensure that a strategy with no active signals results in zero trades."""
    prices = make_prices()
    signals = pd.Series(0, index=prices.index)
    raw_metrics = run_backtest(signals, prices)
    metrics = BacktestMetricsSchema(**_metrics_to_dict(raw_metrics))

    assert metrics.num_trades == 0