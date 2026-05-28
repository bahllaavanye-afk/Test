"""Unit tests for OptionsPCRReversalStrategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.manual.options_pcr_reversal import OptionsPCRReversalStrategy


def _synthetic_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":  close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


def test_strategy_registered():
    from app.strategies import STRATEGY_REGISTRY
    assert "options_pcr_reversal" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["options_pcr_reversal"] is OptionsPCRReversalStrategy


def test_backtest_signals_returns_proper_shape():
    s = OptionsPCRReversalStrategy()
    df = _synthetic_df(200)
    result = s.backtest_signals(df)
    assert len(result.entries) == len(df)
    assert len(result.exits) == len(df)
    assert len(result.short_entries) == len(df)
    assert len(result.short_exits) == len(df)
    assert result.entries.dtype == bool
    assert result.short_entries.dtype == bool


def test_backtest_no_lookahead():
    """First HOLD_BARS+RSI_period bars should never have entry signals (need warmup)."""
    s = OptionsPCRReversalStrategy()
    df = _synthetic_df(200)
    result = s.backtest_signals(df)
    # RSI(2) needs at least 2 bars + shift(1) = no signal in first 3 bars
    assert not result.entries.iloc[:3].any()
    assert not result.short_entries.iloc[:3].any()


def test_short_dataframe_returns_empty_signals():
    s = OptionsPCRReversalStrategy()
    df = _synthetic_df(10)  # too short
    result = s.backtest_signals(df)
    assert not result.entries.any()
    assert not result.short_entries.any()


def test_extreme_oversold_triggers_long():
    """A synthetic crash followed by recovery should produce long entries."""
    s = OptionsPCRReversalStrategy()
    n = 100
    # Build prices that crash then recover so RSI(2) hits below 5 at the bottom
    prices = np.concatenate([
        np.linspace(100, 100, 20),
        np.linspace(100, 70, 10),    # crash
        np.linspace(70, 90, 70),     # recovery
    ])
    df = pd.DataFrame({
        "open": prices, "high": prices * 1.005, "low": prices * 0.995,
        "close": prices, "volume": np.full(n, 1_000_000.0),
    })
    result = s.backtest_signals(df)
    # At least one long entry somewhere in the crash window
    assert result.entries.iloc[20:35].any()


@pytest.mark.asyncio
async def test_analyze_returns_none_without_credentials(monkeypatch):
    """Live analyze() must not fabricate signals when no Alpaca PCR available."""
    s = OptionsPCRReversalStrategy()
    df = _synthetic_df(60)

    # Force PCR fetch to return None (simulating no credentials / API error)
    async def fake_fetch(*args, **kwargs):
        return None
    monkeypatch.setattr(s, "_fetch_pcr", fake_fetch)

    signal = await s.analyze(df, "SPY")
    assert signal is None
