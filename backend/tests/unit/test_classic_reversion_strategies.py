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
import functools
import numpy as np
import pandas as pd
import pytest

from app.strategies import STRATEGY_REGISTRY
from app.strategies.base import BacktestSignals, Signal

NAMES = ["rsi2_pullback", "donchian_breakout", "cci_reversion"]

# ----------------------------------------------------------------------
# Data generation – cached to avoid repeated heavy construction
# ----------------------------------------------------------------------
_df_cache: pd.DataFrame | None = None


def _ohlcv(n: int = 320, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
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


def _get_cached_df() -> pd.DataFrame:
    """Return a single cached OHLCV DataFrame for the test suite."""
    global _df_cache
    if _df_cache is None:
        _df_cache = _ohlcv()
    return _df_cache


# ----------------------------------------------------------------------
# Backtest signal caching – reuse results for the full DataFrame
# ----------------------------------------------------------------------
_backtest_cache: dict[str, BacktestSignals] = {}


def _cached_backtest_signals(name: str, df: pd.DataFrame) -> BacktestSignals:
    """Cache backtest_signals results for a given strategy name and DataFrame."""
    # The DataFrame is immutable for the purposes of the tests, so we can
    # safely cache based solely on the strategy name.
    if name not in _backtest_cache:
        _backtest_cache[name] = STRATEGY_REGISTRY[name]().backtest_signals(df)
    return _backtest_cache[name]


@pytest.mark.parametrize("name", NAMES)
def test_registered(name: str) -> None:
    assert name in STRATEGY_REGISTRY, f"{name} not registered"


@pytest.mark.parametrize("name", NAMES)
def test_backtest_signals_shape(name: str) -> None:
    df = _get_cached_df()
    sig = _cached_backtest_signals(name, df)
    assert isinstance(sig, BacktestSignals)
    assert len(sig.entries) == 320
    assert len(sig.exits) == 320
    # boolean series, no NaN leaking through
    assert sig.entries.dtype == bool
    assert sig.exits.dtype == bool
    assert not sig.entries.isna().any()
    assert not sig.exits.isna().any()


@pytest.mark.parametrize("name", NAMES)
def test_no_entry_on_first_bar(name: str) -> None:
    df = _get_cached_df()
    sig = _cached_backtest_signals(name, df)
    assert not bool(sig.entries.iloc[0]), f"{name}: entry on bar 0 is lookahead bias"


@pytest.mark.parametrize("name", NAMES)
def test_signals_are_causal(name: str) -> None:
    inst = STRATEGY_REGISTRY[name]()
    df = _get_cached_df()
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
def test_analyze_returns_signal_or_none(name: str) -> None:
    inst = STRATEGY_REGISTRY[name]()
    out = asyncio.run(inst.analyze(_get_cached_df(), "TEST"))
    assert out is None or isinstance(out, Signal)
    if isinstance(out, Signal):
        assert out.side in {"buy", "sell"}
        assert 0.0 <= out.confidence <= 1.0
        assert out.strategy_name == name