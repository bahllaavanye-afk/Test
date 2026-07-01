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

# Constants
NUM_BARS = 320
RANDOM_SEED = 7
START_DATE = "2022-01-01"
FREQ = "1D"
RETURN_MEAN = 0.0006
RETURN_STD = 0.015
HIGH_LOW_FACTOR = 0.012
OPEN_STD = 0.004
VOLUME_MIN = 500_000
VOLUME_MAX = 5_000_000
CAUSAL_K_VALUES = (180, 240)
TEST_LABEL = "TEST"
VALID_SIDES = {"buy", "sell"}
STRATEGY_NAMES = ["rsi2_pullback", "donchian_breakout", "cci_reversion"]

def _ohlcv(n: int = NUM_BARS, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    returns = rng.normal(RETURN_MEAN, RETURN_STD, n)
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, HIGH_LOW_FACTOR, n))
    low = close * (1 - rng.uniform(0, HIGH_LOW_FACTOR, n))
    open_ = close * (1 + rng.normal(0, OPEN_STD, n))
    volume = rng.integers(VOLUME_MIN, VOLUME_MAX, n).astype(float)
    idx = pd.date_range(START_DATE, periods=n, freq=FREQ)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_registered(name):
    assert name in STRATEGY_REGISTRY, f"{name} not registered"

@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_backtest_signals_shape(name):
    sig = STRATEGY_REGISTRY[name]().backtest_signals(_ohlcv())
    assert isinstance(sig, BacktestSignals)
    assert len(sig.entries) == NUM_BARS
    assert len(sig.exits) == NUM_BARS
    # boolean series, no NaN leaking through
    assert sig.entries.dtype == bool
    assert sig.exits.dtype == bool
    assert not sig.entries.isna().any()
    assert not sig.exits.isna().any()

@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_no_entry_on_first_bar(name):
    sig = STRATEGY_REGISTRY[name]().backtest_signals(_ohlcv())
    assert not bool(sig.entries.iloc[0]), f"{name}: entry on bar 0 is lookahead bias"

@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_signals_are_causal(name):
    inst = STRATEGY_REGISTRY[name]()
    df = _ohlcv()
    full = inst.backtest_signals(df).entries.reset_index(drop=True)
    for k in CAUSAL_K_VALUES:
        trunc = inst.backtest_signals(df.iloc[:k]).entries.reset_index(drop=True)
        assert len(trunc) == k
        mismatches = int((full.iloc[:k].values != trunc.values).sum())
        assert mismatches == 0, (
            f"{name}: {mismatches} entry(ies) in [0:{k}] changed when future data "
            f"was removed → lookahead bias"
        )

@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_analyze_returns_signal_or_none(name):
    inst = STRATEGY_REGISTRY[name]()
    out = asyncio.run(inst.analyze(_ohlcv(), TEST_LABEL))
    assert out is None or isinstance(out, Signal)
    if isinstance(out, Signal):
        assert out.side in VALID_SIDES
        assert 0.0 <= out.confidence <= 1.0
        assert out.strategy_name == name