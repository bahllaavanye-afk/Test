"""
Unit tests for AnalystRevisionMomentumStrategy.

Covers:
  1. Strategy instantiates with correct attributes
  2. analyze() returns Signal or None (async)
  3. backtest_signals() returns BacktestSignals with correct dtype
  4. No-lookahead: first row is always False
  5. Insufficient data returns None / empty signals
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from app.strategies.base import BacktestSignals, Signal
from app.strategies.manual.analyst_revision_momentum import AnalystRevisionMomentumStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df():
    """200 bars of synthetic daily OHLCV."""
    rng = np.random.default_rng(42)
    n = 200
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def short_df():
    """Only 10 bars — below minimum."""
    rng = np.random.default_rng(7)
    n = 10
    prices = 100 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
    return pd.DataFrame(
        {"close": prices},
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def trending_up_df():
    """200 bars of strongly upward trending prices (should trigger buy)."""
    n = 200
    # Strong uptrend: accelerating over the last 21 bars vs 63 bars
    prices = np.concatenate([
        100 * np.cumprod(1 + np.full(137, 0.001)),        # slow rise
        100 * np.cumprod(1 + np.full(63, 0.001)) * np.cumprod(1 + np.full(137, 0.001))[-1],  # baseline
    ])
    # Reset with faster recent rise
    base = 100 * np.cumprod(1 + np.full(n, 0.001))
    # Accelerate last 21 days
    fast = base.copy()
    fast[-21:] = fast[-22] * np.cumprod(1 + np.full(21, 0.008))
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "close": fast,
            "volume": rng.integers(1_500_000, 3_000_000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


# ---------------------------------------------------------------------------
# Test 1: attributes
# ---------------------------------------------------------------------------

class TestAnalystRevisionMomentumAttributes:
    def test_strategy_name(self):
        assert AnalystRevisionMomentumStrategy().name == "analyst_revision_momentum"

    def test_market_type(self):
        assert AnalystRevisionMomentumStrategy().market_type == "equity"

    def test_strategy_type(self):
        assert AnalystRevisionMomentumStrategy().strategy_type == "manual"

    def test_risk_bucket(self):
        assert AnalystRevisionMomentumStrategy().risk_bucket == "directional"

    def test_default_params(self):
        s = AnalystRevisionMomentumStrategy()
        assert s.SHORT_WINDOW == 21
        assert s.LONG_WINDOW == 63
        assert s.THRESHOLD == pytest.approx(0.25)
        assert s.MIN_BARS == 80

    def test_display_name_not_empty(self):
        assert len(AnalystRevisionMomentumStrategy().display_name) > 0

    def test_confidence_threshold_set(self):
        s = AnalystRevisionMomentumStrategy()
        assert s.confidence_threshold == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Test 2: analyze() — async
# ---------------------------------------------------------------------------

class TestAnalystRevisionMomentumAnalyze:
    def test_analyze_returns_signal_or_none(self, price_df):
        result = asyncio.run(AnalystRevisionMomentumStrategy().analyze(price_df, "AAPL"))
        assert result is None or isinstance(result, Signal)

    def test_analyze_signal_has_direction(self, price_df):
        result = asyncio.run(AnalystRevisionMomentumStrategy().analyze(price_df, "AAPL"))
        if result is not None:
            assert result.side in ("buy", "sell")

    def test_analyze_signal_confidence_in_range(self, price_df):
        result = asyncio.run(AnalystRevisionMomentumStrategy().analyze(price_df, "AAPL"))
        if result is not None:
            assert 0.0 <= result.confidence <= 1.0

    def test_analyze_signal_strategy_name_matches(self, price_df):
        result = asyncio.run(AnalystRevisionMomentumStrategy().analyze(price_df, "AAPL"))
        if result is not None:
            assert result.strategy_name == "analyst_revision_momentum"

    def test_analyze_returns_none_on_insufficient_data(self, short_df):
        result = asyncio.run(AnalystRevisionMomentumStrategy().analyze(short_df, "AAPL"))
        assert result is None

    def test_analyze_returns_none_when_no_close_col(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {"volume": rng.integers(100_000, 1_000_000, n).astype(float)},
            index=pd.date_range("2023-01-01", periods=n, freq="D"),
        )
        result = asyncio.run(AnalystRevisionMomentumStrategy().analyze(df, "AAPL"))
        assert result is None

    def test_analyze_symbol_propagated(self, price_df):
        result = asyncio.run(AnalystRevisionMomentumStrategy().analyze(price_df, "MSFT"))
        if result is not None:
            assert result.symbol == "MSFT"

    def test_analyze_strong_uptrend_may_buy(self, trending_up_df):
        """Strongly accelerating price should produce a buy signal or None (threshold may not be met)."""
        result = asyncio.run(AnalystRevisionMomentumStrategy().analyze(trending_up_df, "AAPL"))
        assert result is None or (isinstance(result, Signal) and result.side == "buy")

    def test_analyze_signal_has_metadata(self, price_df):
        result = asyncio.run(AnalystRevisionMomentumStrategy().analyze(price_df, "AAPL"))
        if result is not None:
            assert "revision_score" in result.metadata
            assert "ret_short_21d" in result.metadata
            assert "ret_long_63d" in result.metadata


# ---------------------------------------------------------------------------
# Test 3: backtest_signals()
# ---------------------------------------------------------------------------

class TestAnalystRevisionMomentumBacktest:
    def test_returns_backtest_signals_type(self, price_df):
        result = AnalystRevisionMomentumStrategy().backtest_signals(price_df)
        assert isinstance(result, BacktestSignals)

    def test_entries_exits_are_bool(self, price_df):
        result = AnalystRevisionMomentumStrategy().backtest_signals(price_df)
        assert result.entries.dtype == bool
        assert result.exits.dtype == bool

    def test_series_length_matches_input(self, price_df):
        n = len(price_df)
        result = AnalystRevisionMomentumStrategy().backtest_signals(price_df)
        assert len(result.entries) == n
        assert len(result.exits) == n

    def test_no_lookahead_first_row_false(self, price_df):
        result = AnalystRevisionMomentumStrategy().backtest_signals(price_df)
        assert not bool(result.entries.iloc[0])

    def test_insufficient_data_returns_all_false(self, short_df):
        # short_df has no close column — but strategy returns empty BacktestSignals
        df = pd.DataFrame(
            {"close": np.linspace(100, 110, 10)},
            index=pd.date_range("2023-01-01", periods=10, freq="D"),
        )
        result = AnalystRevisionMomentumStrategy().backtest_signals(df)
        assert not result.entries.any()
        assert not result.exits.any()

    def test_no_nan_in_signals(self, price_df):
        result = AnalystRevisionMomentumStrategy().backtest_signals(price_df)
        assert not result.entries.isna().any()
        assert not result.exits.isna().any()

    def test_index_matches_input(self, price_df):
        result = AnalystRevisionMomentumStrategy().backtest_signals(price_df)
        pd.testing.assert_index_equal(result.entries.index, price_df.index)
        pd.testing.assert_index_equal(result.exits.index, price_df.index)

    def test_short_entries_present(self, price_df):
        result = AnalystRevisionMomentumStrategy().backtest_signals(price_df)
        assert result.short_entries is not None
        assert len(result.short_entries) == len(price_df)
