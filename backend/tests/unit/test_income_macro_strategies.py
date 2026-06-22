"""Contract tests for income/macro strategies (issues #103, #104, #105).

QA flagged these three as missing unit tests:
  - credit_spread_income   (#105)
  - central_bank_window    (#104)
  - breakeven_inflation    (#103)

Each must register, expose the standard attrs, and return a well-formed
BacktestSignals (bool entries/exits, aligned to the input, no bar-0 lookahead,
and no crash on insufficient data). Pure/offline — synthetic OHLCV only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies import STRATEGY_REGISTRY
from app.strategies.base import BacktestSignals


@pytest.fixture
def daily_ohlcv():
    """300 days of synthetic daily OHLCV."""
    rng = np.random.default_rng(42)
    n = 300
    returns = rng.normal(0.0004, 0.013, n)
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="1D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# name -> expected risk_bucket
_STRATEGIES = {
    "credit_spread_income": "arbitrage",
    "central_bank_window": "directional",
    "breakeven_inflation": "directional",
}


def _get(name):
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    return cls()


@pytest.mark.parametrize("name", list(_STRATEGIES))
def test_registered(name):
    assert name in STRATEGY_REGISTRY


@pytest.mark.parametrize("name,bucket", list(_STRATEGIES.items()))
def test_required_attrs(name, bucket):
    inst = _get(name)
    assert inst.name == name
    assert inst.market_type == "equity"
    assert inst.strategy_type == "manual"
    assert inst.risk_bucket == bucket


@pytest.mark.parametrize("name", list(_STRATEGIES))
def test_backtest_signals_shape(name, daily_ohlcv):
    inst = _get(name)
    sig = inst.backtest_signals(daily_ohlcv)
    assert isinstance(sig, BacktestSignals)
    assert sig.entries.dtype == bool and sig.exits.dtype == bool
    assert len(sig.entries) == len(daily_ohlcv)
    assert len(sig.exits) == len(daily_ohlcv)
    assert not sig.entries.isna().any()
    assert not sig.exits.isna().any()


@pytest.mark.parametrize("name", list(_STRATEGIES))
def test_no_bar0_lookahead(name, daily_ohlcv):
    inst = _get(name)
    sig = inst.backtest_signals(daily_ohlcv)
    assert not bool(sig.entries.iloc[0]), "entry on the first bar is lookahead bias"


@pytest.mark.parametrize("name", list(_STRATEGIES))
def test_insufficient_data_no_crash(name):
    """Too few rows must return empty/aligned signals, never raise."""
    inst = _get(name)
    tiny = pd.DataFrame(
        {"open": [100.0, 101.0], "high": [101.0, 102.0], "low": [99.0, 100.0],
         "close": [100.5, 101.5], "volume": [1e6, 1e6]},
        index=pd.date_range("2023-01-01", periods=2, freq="1D"),
    )
    sig = inst.backtest_signals(tiny)
    assert isinstance(sig, BacktestSignals)
    assert len(sig.entries) == len(tiny)
