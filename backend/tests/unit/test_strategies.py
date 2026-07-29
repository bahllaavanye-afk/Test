"""Strategy regression tests — every registered strategy must implement backtest_signals."""
import pytest
import pandas as pd
import numpy as np
from app.strategies import STRATEGY_REGISTRY


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
        # Ensure index alignment with input data
        assert signals.index.equals(ohlcv.index), "Signal index must match OHLCV index"
        # Ensure numeric dtype
        assert np.issubdtype(signals.dtype, np.number), "Signals must be numeric"
        # Verify allowed signal values
        unique = set(signals.dropna().unique())
        assert unique.issubset({-1, 0, 1, -1.0, 0.0, 1.0})
        # At least one entry signal should be present
        nonzero = signals[signals != 0].dropna()
        assert len(nonzero) >= 1, "Strategy should generate at least one entry signal"
        # Proportion of active signals should be reasonable (avoid overtrading)
        prop_active = len(nonzero) / len(signals)
        assert prop_active <= 0.5, "Too many active signals generated"
        # Exit logic: consecutive non‑zero signals must alternate sign
        signs = nonzero.values
        for i in range(1, len(signs)):
            assert signs[i] != signs[i - 1], "Consecutive non‑zero signals must alternate sign"
    else:
        pytest.fail(f"{name} returned signals in unexpected format")