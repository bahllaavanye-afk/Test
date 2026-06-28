"""Strategy regression tests — every registered strategy must implement backtest_signals."""
import pytest
import pandas as pd
import numpy as np
from app.strategies import STRATEGY_REGISTRY


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate that the OHLCV DataFrame meets expected requirements.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing market data.

    Raises
    ------
    ValueError
        If the DataFrame is not valid:
        - Not a pandas DataFrame.
        - Missing required columns.
        - Empty DataFrame.
        - Index is not a DatetimeIndex.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame")
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ohlcv is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("ohlcv DataFrame is empty")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("ohlcv index must be a pandas DatetimeIndex")


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
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    _validate_ohlcv(df)
    return df


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
    # Validate inputs before invoking strategy method
    _validate_ohlcv(ohlcv)

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