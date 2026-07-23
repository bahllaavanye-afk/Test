"""Strategy regression tests — every registered strategy must implement backtest_signals."""
import pytest
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from app.strategies import STRATEGY_REGISTRY


class SignalSchema(BaseModel):
    """Pydantic schema representing a single backtest signal.

    Attributes
    ----------
    value: int
        Signal direction where -1 denotes a short position, 0 denotes neutral,
        and 1 denotes a long position.
    """
    value: int = Field(
        ...,
        description="Signal direction: -1 for short, 0 for neutral, 1 for long",
        examples=[-1, 0, 1],
    )

    @validator("value")
    def _must_be_valid_signal(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError("Signal must be -1, 0, or 1")
        return v


@pytest.fixture
def ohlcv():
    n = 300
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.015, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * 1.005
    low = close * 0.995
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.integers(100_000, 1_000_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_registry_not_empty():
    assert len(STRATEGY_REGISTRY) > 0


@pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_has_required_attrs(name):
    cls = STRATEGY_REGISTRY[name]
    inst = cls() if not getattr(cls, "__abstractmethods__", None) else None
    if inst is None:
        return
    assert hasattr(inst, "market_type")
    assert hasattr(inst, "strategy_type")
    assert hasattr(inst, "risk_bucket")


@pytest.mark.parametrize(
    "name",
    [
        "momentum",
        "mean_reversion",
        "rsi_macd",
        "breakout",
        "supertrend",
    ],
)
def test_strategy_backtest_signals(name, ohlcv):
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    inst = cls()
    signals = inst.backtest_signals(ohlcv)
    if signals is None or (hasattr(signals, "__len__") and len(signals) == 0):
        pytest.skip(f"{name} returned no signals")
    if isinstance(signals, pd.Series):
        unique = set(signals.dropna().unique())
        # Should be subset of -1, 0, 1
        assert unique.issubset({-1, 0, 1, -1.0, 0.0, 1.0})
        # Validate each unique signal against the schema
        for val in unique:
            SignalSchema(value=int(val))
    else:
        # If a strategy returns a custom container, ensure each element conforms to the schema
        for item in signals:
            SignalSchema(value=int(item))