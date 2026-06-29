"""Tests for the classic reversion/breakout strategies (#alpha-research).

Covers rsi2_pullback, donchian_breakout, cci_reversion:
  * they load from STRATEGY_REGISTRY,
  * backtest_signals() returns a boolean-entry/exit BacktestSignals,
  * no entry on bar 0 (that would be lookahead), and
  * signals are causal — entries on df[:k] exactly match entries[:k] on the full
    series (truncating away the future can't change the past).
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from app.strategies import STRATEGY_REGISTRY
from app.strategies.base import BacktestSignals, Signal

NAMES = ["rsi2_pullback", "donchian_breakout", "cci_reversion"]


def _ohlcv(n=320, seed=7):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0006, 0.015, n)
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


@pytest.mark.parametrize("name", NAMES)
def test_registered(name):
    assert name in STRATEGY_REGISTRY, f"{name} not registered"


@pytest.mark.parametrize("name", NAMES)
def test_backtest_signals_shape(name):
    sig = STRATEGY_REGISTRY[name]().backtest_signals(_ohlcv())
    assert isinstance(sig, BacktestSignals)
    assert len(sig.entries) == 320
    assert len(sig.exits) == 320
    # boolean series, no NaN leaking through
    assert sig.entries.dtype == bool
    assert sig.exits.dtype == bool
    assert not sig.entries.isna().any()
    assert not sig.exits.isna().any()


@pytest.mark.parametrize("name", NAMES)
def test_no_entry_on_first_bar(name):
    sig = STRATEGY_REGISTRY[name]().backtest_signals(_ohlcv())
    assert not bool(sig.entries.iloc[0]), f"{name}: entry on bar 0 is lookahead bias"


@pytest.mark.parametrize("name", NAMES)
def test_signals_are_causal(name):
    inst = STRATEGY_REGISTRY[name]()
    df = _ohlcv()
    full = inst.backtest_signals(df).entries.reset_index(drop=True)
    for k in (180, 240):
        trunc = inst.backtest_signals(df.iloc[:k]).entries.reset_index(drop=True)
        assert len(trunc) == k
        mismatches = int((full.iloc[:k].values != trunc.values).sum())
        assert mismatches == 0, (
            f"{name}: {mismatches} entry(ies) in [0:{k}] changed when future data "
            f"was removed → lookahead bias"
        )


@pytest.mark.parametrize("name", NAMES)
def test_analyze_returns_signal_or_none(name):
    inst = STRATEGY_REGISTRY[name]()
    out = asyncio.run(inst.analyze(_ohlcv(), "TEST"))
    assert out is None or isinstance(out, Signal)
    if isinstance(out, Signal):
        assert out.side in {"buy", "sell"}
        assert 0.0 <= out.confidence <= 1.0
        assert out.strategy_name == name
