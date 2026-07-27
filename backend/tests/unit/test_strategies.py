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
    ["momentum", "mean_reversion", "rsi_macd", "breakout", "supertrend"],
)
def test_strategy_backtest_signals(name, ohlcv):
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    inst = cls()
    signals = inst.backtest_signals(ohlcv)

    # Skip strategies that legitimately produce no signals for the test data
    if signals is None or (hasattr(signals, "__len__") and len(signals) == 0):
        pytest.skip(f"{name} returned no signals")

    # Normalize to pandas Series for further checks
    if isinstance(signals, pd.DataFrame):
        # Expect a single column named 'signal' if DataFrame is used
        if signals.shape[1] != 1:
            pytest.fail(f"{name} returned DataFrame with multiple columns")
        signals = signals.iloc[:, 0]

    assert isinstance(signals, pd.Series), f"{name} must return a pandas Series"

    # Index alignment
    assert signals.index.equals(ohlcv.index), f"{name} index mismatch with input OHLCV"

    # Value domain
    unique = set(signals.dropna().unique())
    assert unique.issubset({-1, 0, 1, -1.0, 0.0, 1.0}), f"{name} contains invalid signal values"

    # Basic signal quality checks
    # 1. Ensure we have at least one long and one short signal for bidirectional strategies
    if name in {"momentum", "mean_reversion", "rsi_macd", "breakout"}:
        assert -1 in unique or -1.0 in unique, f"{name} missing short signal"
        assert 1 in unique or 1.0 in unique, f"{name} missing long signal"

    # 2. No consecutive non‑zero signals without an intervening flat (0) signal
    non_zero = signals != 0
    consecutive = non_zero & non_zero.shift(1).fillna(False)
    assert not consecutive.any(), f"{name} has consecutive non‑zero signals without a flat"

    # 3. Minimum holding period: after a position change, enforce at least 2 periods before reversal
    min_holding = 2
    position_changes = signals.diff().abs() > 0
    change_indices = np.where(position_changes)[0]
    if len(change_indices) > 1:
        intervals = np.diff(change_indices)
        assert (intervals >= min_holding).all(), f"{name} reverses positions too quickly"


def test_signal_series_properties(name="momentum", ohlcv=ohlcv()):
    """
    Additional sanity check for a representative strategy.
    Verifies that the signal series respects typical entry/exit constraints:
    - Entry signals are only generated when the previous bar was flat (0).
    - Exit signals (reversal to flat) are only issued after a minimum holding period.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    inst = cls()
    signals = inst.backtest_signals(ohlcv)
    if signals is None or (hasattr(signals, "__len__") and len(signals) == 0):
        pytest.skip(f"{name} returned no signals")
    if isinstance(signals, pd.DataFrame):
        signals = signals.iloc[:, 0]

    # Ensure entry occurs from flat state
    entry_mask = signals != 0
    prev_flat = signals.shift(1).fillna(0) == 0
    assert (entry_mask & ~prev_flat).sum() == 0, f"{name} generated entry without prior flat"

    # Enforce minimum holding period before exiting to flat
    min_hold = 2
    positions = signals.replace(0, np.nan).ffill()
    exit_mask = signals == 0
    exit_positions = positions[exit_mask]
    # Compute how many periods each position lasted
    durations = exit_positions.groupby((exit_positions != exit_positions.shift()).cumsum()).apply(
        lambda grp: len(grp)
    )
    if not durations.empty:
        assert (durations >= min_hold).all(), f"{name} exited positions shorter than minimum holding period"