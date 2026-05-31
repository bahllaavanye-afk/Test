"""Unit tests for TimeSeriesMomentumStrategy (Moskowitz-Ooi-Pedersen 2012)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.manual.time_series_momentum import TimeSeriesMomentumStrategy


def _df_trending(direction: int = 1, n: int = 300) -> pd.DataFrame:
    """Synthetic strongly trending series — strong enough drift to dominate noise."""
    rng = np.random.default_rng(7)
    drift = direction * 0.003  # ~75% annual drift, large enough to dominate seed-noise
    rets = rng.normal(drift, 0.008, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(n, 1_000_000.0),
    })


def test_registered():
    from app.strategies import STRATEGY_REGISTRY
    assert "time_series_momentum" in STRATEGY_REGISTRY


def test_backtest_signal_shape():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=300)
    out = s.backtest_signals(df)
    assert len(out.entries) == len(df)
    assert out.entries.dtype == bool
    assert out.short_entries.dtype == bool


def test_uptrend_produces_long_entries():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=300)
    out = s.backtest_signals(df)
    # After warmup, long entries should appear; no short entries (it's uptrending)
    assert out.entries.iloc[-30:].any()
    assert not out.short_entries.iloc[-30:].any()


def test_downtrend_produces_short_entries():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=-1, n=300)
    out = s.backtest_signals(df)
    assert out.short_entries.iloc[-30:].any()
    assert not out.entries.iloc[-30:].any()


def test_no_lookahead_in_warmup():
    """The first `lookback` bars must produce no signals — need 252 bars of history."""
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=300)
    out = s.backtest_signals(df)
    assert not out.entries.iloc[:s.lookback].any()
    assert not out.short_entries.iloc[:s.lookback].any()


def test_short_data_returns_empty():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=50)  # less than lookback
    out = s.backtest_signals(df)
    assert not out.entries.any()
    assert not out.short_entries.any()


@pytest.mark.asyncio
async def test_analyze_uptrend_returns_buy():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=300)
    sig = await s.analyze(df, "SPY")
    assert sig is not None
    assert sig.side == "buy"
    assert 0 < sig.confidence <= 0.95
    assert "ret_12m" in sig.metadata
    assert sig.metadata["ret_12m"] > 0


@pytest.mark.asyncio
async def test_analyze_downtrend_returns_sell():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=-1, n=300)
    sig = await s.analyze(df, "QQQ")
    assert sig is not None
    assert sig.side == "sell"
    assert sig.metadata["ret_12m"] < 0


@pytest.mark.asyncio
async def test_analyze_returns_none_on_short_data():
    s = TimeSeriesMomentumStrategy()
    df = _df_trending(direction=1, n=50)
    sig = await s.analyze(df, "MSFT")
    assert sig is None
