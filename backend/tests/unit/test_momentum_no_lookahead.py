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


def _ohlcv(n=320, seed=11):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.012, n))
    low = close * (1 - rng.uniform(0, 0.012, n))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="1D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


# Single-symbol momentum strategies that consume plain OHLCV and return BacktestSignals.
_NAMES = [
    "momentum",
    "time_series_momentum",
    "micro_cap_momentum",
    "triple_barrier_momentum",
    "crypto_whale_momentum",
]


def _entries(sig):
    return sig.entries if isinstance(sig, BacktestSignals) else (sig > 0)


@pytest.mark.parametrize("name", _NAMES)
def test_momentum_signals_are_causal(name):
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
        # entries[:k] must match exactly — no future bar may influence a past signal
        mismatches = int((full.iloc[:k].values != trunc.values).sum())
        assert mismatches == 0, (
            f"{name}: {mismatches} entry(ies) in [0:{k}] changed when future data "
            f"was removed → lookahead bias"
        )


@pytest.mark.parametrize("name", _NAMES)
def test_momentum_no_entry_on_first_bar(name):
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    sig = cls().backtest_signals(_ohlcv())
    assert not bool(_entries(sig).iloc[0]), f"{name}: entry on bar 0 is lookahead bias"
