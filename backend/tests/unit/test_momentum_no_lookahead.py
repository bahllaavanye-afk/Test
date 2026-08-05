"""Lookahead-bias guard for momentum strategies (#alpha-research).

The strategies already shift signals by 1 bar; this locks that in. The gold
standard: a signal at bar i may only depend on data up to bar i. So entries
computed on the full series, restricted to [:k], must EXACTLY equal entries
computed on the truncated series df[:k] — truncating away the future can't
change the past. Any mismatch is lookahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies import STRATEGY_REGISTRY
from app.strategies.base import BacktestSignals

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _ohlcv(n: int = 320, seed: int = 11) -> pd.DataFrame:
    """Generate a deterministic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.012, n))
    low = close * (1 - rng.uniform(0, 0.012, n))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="1D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _entries(sig):
    """Extract entry boolean series from a BacktestSignals object."""
    return sig.entries if isinstance(sig, BacktestSignals) else (sig > 0)


def _exits(sig):
    """Extract exit boolean series from a BacktestSignals object."""
    return sig.exits if isinstance(sig, BacktestSignals) else (sig < 0)


def _momentum(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Simple time‑series momentum: pct change over `window` periods."""
    return df["close"].pct_change(window)


# --------------------------------------------------------------------------- #
# Test suite
# --------------------------------------------------------------------------- #
_NAMES = [
    "momentum",
    "time_series_momentum",
    "micro_cap_momentum",
    "triple_barrier_momentum",
    "crypto_whale_momentum",
]


@pytest.mark.parametrize("name", _NAMES)
def test_momentum_signals_are_causal(name):
    """Ensure no look‑ahead bias: truncating future data must not alter past signals."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    inst = cls()
    df = _ohlcv()

    full = _entries(inst.backtest_signals(df)).reset_index(drop=True)

    # Truncate away the future at two checkpoints; the past must be unchanged.
    for k in (180, 240):
        trunc = _entries(inst.backtest_signals(df.iloc[:k])).reset_index(drop=True)
        assert len(trunc) == k
        mismatches = int((full.iloc[:k].values != trunc.values).sum())
        assert mismatches == 0, (
            f"{name}: {mismatches} entry(ies) in [0:{k}] changed when future data "
            f"was removed → lookahead bias"
        )


@pytest.mark.parametrize("name", _NAMES)
def test_momentum_no_entry_on_first_bar(name):
    """First bar should never generate an entry (no prior data available)."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    sig = cls().backtest_signals(_ohlcv())
    assert not bool(_entries(sig).iloc[0]), f"{name}: entry on bar 0 is lookahead bias"


@pytest.mark.parametrize("name", _NAMES)
def test_momentum_entry_filters(name):
    """Entries must satisfy basic momentum and volume confirmation filters."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    df = _ohlcv()
    sig = cls().backtest_signals(df)

    entries = _entries(sig)
    momentum = _momentum(df, window=5)
    median_vol = df["volume"].median()

    # Iterate over entry points and validate filters.
    for idx in np.where(entries.values)[0]:
        # Positive momentum over the look‑back window
        assert momentum.iloc[idx] > 0, (
            f"{name}: entry at bar {idx} occurs with non‑positive momentum "
            f"({momentum.iloc[idx]:.4f})"
        )
        # Volume should be above the median to avoid low‑liquidity entries
        assert df["volume"].iloc[idx] > median_vol, (
            f"{name}: entry at bar {idx} occurs on low volume "
            f"({df['volume'].iloc[idx]:.0f} ≤ median {median_vol:.0f})"
        )


@pytest.mark.parametrize("name", _NAMES)
def test_momentum_exit_logic(name):
    """Validate that exits occur within a reasonable holding period or when momentum reverses."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    df = _ohlcv()
    sig = cls().backtest_signals(df)

    entries = _entries(sig)
    exits = _exits(sig)
    momentum = _momentum(df, window=5)

    max_holding = 20  # bars

    entry_indices = np.where(entries.values)[0]
    exit_indices = np.where(exits.values)[0]

    for entry_idx in entry_indices:
        # Find the first exit after the entry
        subsequent_exits = exit_indices[exit_indices > entry_idx]
        if subsequent_exits.size == 0:
            # No exit recorded – acceptable if still holding at end of data
            continue
        exit_idx = subsequent_exits[0]

        # Holding period constraint
        holding = exit_idx - entry_idx
        assert holding <= max_holding, (
            f"{name}: holding period of {holding} bars exceeds max {max_holding} "
            f"from entry {entry_idx} to exit {exit_idx}"
        )

        # Momentum reversal check: exit should happen when momentum turns non‑positive
        assert momentum.iloc[exit_idx] <= 0, (
            f"{name}: exit at bar {exit_idx} occurs while momentum remains positive "
            f"({momentum.iloc[exit_idx]:.4f})"
        )