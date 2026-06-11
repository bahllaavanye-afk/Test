"""
Unit tests for the Avellaneda-Stoikov Market Making strategy.

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
from app.strategies.manual.avellaneda_stoikov_mm import AvellanedaStoikovMM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df():
    """200 bars of synthetic 1-minute OHLCV."""
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
        index=pd.date_range("2023-01-01", periods=n, freq="1min"),
    )


@pytest.fixture
def short_df():
    """Only 5 bars — below minimum."""
    rng = np.random.default_rng(7)
    n = 5
    prices = 100 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
    return pd.DataFrame(
        {"close": prices},
        index=pd.date_range("2023-01-01", periods=n, freq="1min"),
    )


# ---------------------------------------------------------------------------
# Test 1: name and attributes
# ---------------------------------------------------------------------------

class TestAvellanedaStoikovMMAttributes:
    def test_strategy_name(self):
        assert AvellanedaStoikovMM().name == "avellaneda_stoikov_mm"

    def test_market_type(self):
        assert AvellanedaStoikovMM().market_type == "crypto"

    def test_strategy_type(self):
        assert AvellanedaStoikovMM().strategy_type == "manual"

    def test_risk_bucket(self):
        assert AvellanedaStoikovMM().risk_bucket == "arbitrage"

    def test_default_params(self):
        s = AvellanedaStoikovMM()
        assert s.gamma == pytest.approx(0.1)
        assert s.kappa == pytest.approx(1.5)
        assert s.T == pytest.approx(300.0)

    def test_custom_params(self):
        s = AvellanedaStoikovMM(params={"gamma": 0.2, "kappa": 2.0, "T": 600.0})
        assert s.gamma == pytest.approx(0.2)
        assert s.kappa == pytest.approx(2.0)
        assert s.T == pytest.approx(600.0)

    def test_display_name_not_empty(self):
        assert len(AvellanedaStoikovMM().display_name) > 0

    def test_description_returns_string(self):
        desc = AvellanedaStoikovMM().description()
        assert isinstance(desc, str) and len(desc) > 0


# ---------------------------------------------------------------------------
# Test 2: analyze() — async
# ---------------------------------------------------------------------------

class TestAvellanedaStoikovMMAnalyze:
    def test_analyze_returns_signal_or_none(self, price_df):
        result = asyncio.run(AvellanedaStoikovMM().analyze(price_df, "BTCUSDT"))
        assert result is None or isinstance(result, Signal)

    def test_analyze_signal_has_direction(self, price_df):
        result = asyncio.run(AvellanedaStoikovMM().analyze(price_df, "BTCUSDT"))
        if result is not None:
            assert result.side in ("buy", "sell")

    def test_analyze_signal_confidence_in_range(self, price_df):
        result = asyncio.run(AvellanedaStoikovMM().analyze(price_df, "BTCUSDT"))
        if result is not None:
            assert 0.0 <= result.confidence <= 1.0

    def test_analyze_signal_strategy_name_matches(self, price_df):
        result = asyncio.run(AvellanedaStoikovMM().analyze(price_df, "BTCUSDT"))
        if result is not None:
            assert result.strategy_name == "avellaneda_stoikov_mm"

    def test_analyze_returns_none_on_insufficient_data(self, short_df):
        result = asyncio.run(AvellanedaStoikovMM().analyze(short_df, "BTCUSDT"))
        assert result is None

    def test_analyze_returns_none_on_none_data(self):
        result = asyncio.run(AvellanedaStoikovMM().analyze(None, "BTCUSDT"))
        assert result is None

    def test_analyze_returns_none_on_empty_df(self):
        df = pd.DataFrame({"close": []})
        result = asyncio.run(AvellanedaStoikovMM().analyze(df, "BTCUSDT"))
        assert result is None

    def test_analyze_symbol_propagated(self, price_df):
        result = asyncio.run(AvellanedaStoikovMM().analyze(price_df, "ETHUSDT"))
        if result is not None:
            assert result.symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# Test 3: backtest_signals()
# ---------------------------------------------------------------------------

class TestAvellanedaStoikovMMBacktest:
    def test_returns_backtest_signals_type(self, price_df):
        result = AvellanedaStoikovMM().backtest_signals(price_df)
        assert isinstance(result, BacktestSignals)

    def test_entries_exits_are_bool(self, price_df):
        result = AvellanedaStoikovMM().backtest_signals(price_df)
        assert result.entries.dtype == bool
        assert result.exits.dtype == bool

    def test_series_length_matches_input(self, price_df):
        n = len(price_df)
        result = AvellanedaStoikovMM().backtest_signals(price_df)
        assert len(result.entries) == n
        assert len(result.exits) == n

    def test_no_lookahead_first_row_false(self, price_df):
        result = AvellanedaStoikovMM().backtest_signals(price_df)
        assert not bool(result.entries.iloc[0])

    def test_insufficient_data_returns_all_false(self, short_df):
        result = AvellanedaStoikovMM().backtest_signals(short_df)
        assert not result.entries.any()
        assert not result.exits.any()

    def test_index_matches_input(self, price_df):
        result = AvellanedaStoikovMM().backtest_signals(price_df)
        pd.testing.assert_index_equal(result.entries.index, price_df.index)
        pd.testing.assert_index_equal(result.exits.index, price_df.index)

    def test_no_nan_in_signals(self, price_df):
        result = AvellanedaStoikovMM().backtest_signals(price_df)
        assert not result.entries.isna().any()
        assert not result.exits.isna().any()
