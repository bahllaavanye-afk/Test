"""Unit tests for TimeSeriesMomentumStrategy (Moskowitz-Ooi-Pedersen 2012)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.manual.time_series_momentum import TimeSeriesMomentumStrategy

# Named constants
DIRECTION_UP = 1
DIRECTION_DOWN = -1
DEFAULT_N = 300
DEFAULT_DRIFT = 0.003
DEFAULT_VOLATILITY = 0.008
DEFAULT_PRICE = 100
DEFAULT_VOLUME = 1_000_000.0
DEFAULT_WARMUP = 30
DEFAULT_LOOKBACK = 252
DEFAULT_confidence_THRESHOLD = 0.95
DEFAULT_RET_12M_KEY = "ret_12m"
DEFAULT_SIDE_BUY = "buy"
DEFAULT_SIDE_SELL = "sell"

def _df_trending(direction: int = DIRECTION_UP, n: int = DEFAULT_N) -> pd.DataFrame:
    """Synthetic strongly trending series — strong enough drift to dominate noise."""
    rng = np.random.default_rng(7)
    drift = direction * DEFAULT_DRIFT  # ~75% annual drift, large enough to dominate seed-noise
    rets = rng.normal(drift, DEFAULT_VOLATILITY, n)
    close = DEFAULT_PRICE * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(n, DEFAULT_VOLUME),
    })


def test_registered():
    from app.strategies import STRATEGY_REGISTRY
    assert "time_series_momentum" in STRATEGY_REGISTRY


def test_backtest_signal_shape():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=DIRECTION_UP, n=DEFAULT_N)
    out = s.backtest_signals(df)
    assert len(out.entries) == len(df)
    assert out.entries.dtype == bool
    assert out.short_entries.dtype == bool


def test_uptrend_produces_long_entries():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=DIRECTION_UP, n=DEFAULT_N)
    out = s.backtest_signals(df)
    # After warmup, long entries should appear; no short entries (it's uptrending)
    assert out.entries.iloc[-DEFAULT_WARMUP:].any()
    assert not out.short_entries.iloc[-DEFAULT_WARMUP:].any()


def test_downtrend_produces_short_entries():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=DIRECTION_DOWN, n=DEFAULT_N)
    out = s.backtest_signals(df)
    assert out.short_entries.iloc[-DEFAULT_WARMUP:].any()
    assert not out.entries.iloc[-DEFAULT_WARMUP:].any()


def test_no_lookahead_in_warmup():
    """The first `lookback` bars must produce no signals — need 252 bars of history."""
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=DIRECTION_UP, n=DEFAULT_N)
    out = s.backtest_signals(df)
    assert not out.entries.iloc[:s.lookback].any()
    assert not out.short_entries.iloc[:s.lookback].any()


def test_short_data_returns_empty():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=DIRECTION_UP, n=50)  # less than lookback
    out = s.backtest_signals(df)
    assert not out.entries.any()
    assert not out.short_entries.any()


@pytest.mark.asyncio
async def test_analyze_uptrend_returns_buy():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=DIRECTION_UP, n=DEFAULT_N)
    sig = await s.analyze(df, "SPY")
    assert sig is not None
    assert sig.side == DEFAULT_SIDE_BUY
    assert 0 < sig.confidence <= DEFAULT_confidence_THRESHOLD
    assert DEFAULT_RET_12M_KEY in sig.metadata
    assert sig.metadata[DEFAULT_RET_12M_KEY] > 0


@pytest.mark.asyncio
async def test_analyze_downtrend_returns_sell():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=DIRECTION_DOWN, n=DEFAULT_N)
    sig = await s.analyze(df, "QQQ")
    assert sig is not None
    assert sig.side == DEFAULT_SIDE_SELL
    assert sig.metadata[DEFAULT_RET_12M_KEY] < 0


@pytest.mark.asyncio
async def test_analyze_returns_none_on_short_data():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=DIRECTION_UP, n=50)
    sig = await s.analyze(df, "MSFT")
    assert sig is None