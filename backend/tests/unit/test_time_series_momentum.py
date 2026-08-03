"""Unit tests for TimeSeriesMomentumStrategy (Moskowitz-Ooi-Pedersen 2012)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.manual.time_series_momentum import TimeSeriesMomentumStrategy

# Constants
SEED = 7
DEFAULT_DRIFT = 0.003
DEFAULT_NOISE = 0.008
DEFAULT_PRICE_SCALE = 100
HIGH_MULTIPLIER = 1.01
LOW_MULTIPLIER = 0.99
DEFAULT_VOLUME = 1_000_000.0
DEFAULT_TREND_N = 300
SHORT_TREND_N = 50
LOOKBACK_CHECK_BARS = 30
MAX_CONFIDENCE = 0.95
REGISTRY_KEY = "time_series_momentum"
SYMBOL_SPY = "SPY"
SYMBOL_QQQ = "QQQ"
SYMBOL_MSFT = "MSFT"
SIDE_BUY = "buy"
SIDE_SELL = "sell"
METADATA_RET_12M = "ret_12m"


def _df_trending(direction: int = 1, n: int = DEFAULT_TREND_N) -> pd.DataFrame:
    """Synthetic strongly trending series — strong enough drift to dominate noise."""
    rng = np.random.default_rng(SEED)
    drift = direction * DEFAULT_DRIFT  # ~75% annual drift, large enough to dominate seed-noise
    rets = rng.normal(drift, DEFAULT_NOISE, n)
    close = DEFAULT_PRICE_SCALE * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open": close,
        "high": close * HIGH_MULTIPLIER,
        "low": close * LOW_MULTIPLIER,
        "close": close,
        "volume": np.full(n, DEFAULT_VOLUME),
    })


def test_registered():
    from app.strategies import STRATEGY_REGISTRY
    assert REGISTRY_KEY in STRATEGY_REGISTRY


def test_backtest_signal_shape():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=DEFAULT_TREND_N)
    out = s.backtest_signals(df)
    assert len(out.entries) == len(df)
    assert out.entries.dtype == bool
    assert out.short_entries.dtype == bool


def test_uptrend_produces_long_entries():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=DEFAULT_TREND_N)
    out = s.backtest_signals(df)
    # After warmup, long entries should appear; no short entries (it's uptrending)
    assert out.entries.iloc[-LOOKBACK_CHECK_BARS:].any()
    assert not out.short_entries.iloc[-LOOKBACK_CHECK_BARS:].any()


def test_downtrend_produces_short_entries():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=-1, n=DEFAULT_TREND_N)
    out = s.backtest_signals(df)
    assert out.short_entries.iloc[-LOOKBACK_CHECK_BARS:].any()
    assert not out.entries.iloc[-LOOKBACK_CHECK_BARS:].any()


def test_no_lookahead_in_warmup():
    """The first `lookback` bars must produce no signals — need 252 bars of history."""
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=DEFAULT_TREND_N)
    out = s.backtest_signals(df)
    assert not out.entries.iloc[:s.lookback].any()
    assert not out.short_entries.iloc[:s.lookback].any()


def test_short_data_returns_empty():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=SHORT_TREND_N)  # less than lookback
    out = s.backtest_signals(df)
    assert not out.entries.any()
    assert not out.short_entries.any()


@pytest.mark.asyncio
async def test_analyze_uptrend_returns_buy():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=DEFAULT_TREND_N)
    sig = await s.analyze(df, SYMBOL_SPY)
    assert sig is not None
    assert sig.side == SIDE_BUY
    assert 0 < sig.confidence <= MAX_CONFIDENCE
    assert METADATA_RET_12M in sig.metadata
    assert sig.metadata[METADATA_RET_12M] > 0


@pytest.mark.asyncio
async def test_analyze_downtrend_returns_sell():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=-1, n=DEFAULT_TREND_N)
    sig = await s.analyze(df, SYMBOL_QQQ)
    assert sig is not None
    assert sig.side == SIDE_SELL
    assert sig.metadata[METADATA_RET_12M] < 0


@pytest.mark.asyncio
async def test_analyze_returns_none_on_short_data():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=SHORT_TREND_N)
    sig = await s.analyze(df, SYMBOL_MSFT)
    assert sig is None