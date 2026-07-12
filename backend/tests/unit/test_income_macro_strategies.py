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

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
RANDOM_SEED = 42
NUM_DAYS = 300
RETURN_MEAN = 0.0004
RETURN_STD = 0.013
INITIAL_CLOSE = 100.0
HIGH_UNIFORM_MAX = 0.01
LOW_UNIFORM_MAX = 0.01
OPEN_NORMAL_STD = 0.003
VOLUME_MIN = 500_000
VOLUME_MAX = 5_000_000

DATE_START = "2023-01-01"
DATE_FREQ = "1D"

RISK_BUCKET_ARBITRAGE = "arbitrage"
RISK_BUCKET_DIRECTIONAL = "directional"
MARKET_TYPE_EQUITY = "equity"
STRATEGY_TYPE_MANUAL = "manual"

LOOKAHEAD_ERROR_MSG = "entry on the first bar is lookahead bias"

# Tiny synthetic dataset constants
TINY_OPEN = [100.0, 101.0]
TINY_HIGH = [101.0, 102.0]
TINY_LOW = [99.0, 100.0]
TINY_CLOSE = [100.5, 101.5]
TINY_VOLUME = [1_000_000, 1_000_000]
TINY_PERIODS = 2
TINY_DATE_START = "2023-01-01"
TINY_DATE_FREQ = "1D"

# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
@pytest.fixture
def daily_ohlcv():
    """Synthetic daily OHLCV for NUM_DAYS days."""
    rng = np.random.default_rng(RANDOM_SEED)
    returns = rng.normal(RETURN_MEAN, RETURN_STD, NUM_DAYS)
    close = INITIAL_CLOSE * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, HIGH_UNIFORM_MAX, NUM_DAYS))
    low = close * (1 - rng.uniform(0, LOW_UNIFORM_MAX, NUM_DAYS))
    open_ = close * (1 + rng.normal(0, OPEN_NORMAL_STD, NUM_DAYS))
    volume = rng.integers(VOLUME_MIN, VOLUME_MAX, NUM_DAYS).astype(float)
    idx = pd.date_range(DATE_START, periods=NUM_DAYS, freq=DATE_FREQ)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# name -> expected risk_bucket
_STRATEGIES = {
    "credit_spread_income": RISK_BUCKET_ARBITRAGE,
    "central_bank_window": RISK_BUCKET_DIRECTIONAL,
    "breakeven_inflation": RISK_BUCKET_DIRECTIONAL,
}


def _get(name):
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        pytest.skip(f"{name} not in registry")
    return cls()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", list(_STRATEGIES))
def test_registered(name):
    assert name in STRATEGY_REGISTRY


@pytest.mark.parametrize("name,bucket", list(_STRATEGIES.items()))
def test_required_attrs(name, bucket):
    inst = _get(name)
    assert inst.name == name
    assert inst.market_type == MARKET_TYPE_EQUITY
    assert inst.strategy_type == STRATEGY_TYPE_MANUAL
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
    assert not bool(sig.entries.iloc[0]), LOOKAHEAD_ERROR_MSG


@pytest.mark.parametrize("name", list(_STRATEGIES))
def test_insufficient_data_no_crash(name):
    """Too few rows must return empty/aligned signals, never raise."""
    inst = _get(name)
    tiny = pd.DataFrame(
        {
            "open": TINY_OPEN,
            "high": TINY_HIGH,
            "low": TINY_LOW,
            "close": TINY_CLOSE,
            "volume": TINY_VOLUME,
        },
        index=pd.date_range(TINY_DATE_START, periods=TINY_PERIODS, freq=TINY_DATE_FREQ),
    )
    sig = inst.backtest_signals(tiny)
    assert isinstance(sig, BacktestSignals)
    assert len(sig.entries) == len(tiny)