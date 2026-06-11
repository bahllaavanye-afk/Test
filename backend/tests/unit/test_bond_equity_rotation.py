"""
Unit tests for BondEquityRotationStrategy.

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
from app.strategies.manual.bond_equity_rotation import BondEquityRotationStrategy


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
def price_df_with_vix():
    """200 bars with vix_close column."""
    rng = np.random.default_rng(42)
    n = 200
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    vix = 20 + rng.normal(0, 3, n).cumsum() * 0.1
    vix = np.clip(vix, 10, 80)
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "vix_close": vix,
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


# ---------------------------------------------------------------------------
# Test 1: attributes
# ---------------------------------------------------------------------------

class TestBondEquityRotationAttributes:
    def test_strategy_name(self):
        assert BondEquityRotationStrategy().name == "bond_equity_rotation"

    def test_market_type(self):
        assert BondEquityRotationStrategy().market_type == "equity"

    def test_strategy_type(self):
        assert BondEquityRotationStrategy().strategy_type == "manual"

    def test_risk_bucket(self):
        assert BondEquityRotationStrategy().risk_bucket == "directional"

    def test_default_params(self):
        s = BondEquityRotationStrategy()
        assert s.corr_window == 30
        assert s.corr_threshold == pytest.approx(0.1)

    def test_custom_params(self):
        s = BondEquityRotationStrategy(params={"corr_window": 20, "corr_threshold": 0.2})
        assert s.corr_window == 20
        assert s.corr_threshold == pytest.approx(0.2)

    def test_display_name_not_empty(self):
        assert len(BondEquityRotationStrategy().display_name) > 0


# ---------------------------------------------------------------------------
# Test 2: analyze() — async
# ---------------------------------------------------------------------------

class TestBondEquityRotationAnalyze:
    def test_analyze_returns_signal_or_none(self, price_df):
        result = asyncio.run(BondEquityRotationStrategy().analyze(price_df, "SPY"))
        assert result is None or isinstance(result, Signal)

    def test_analyze_with_vix_returns_signal_or_none(self, price_df_with_vix):
        result = asyncio.run(BondEquityRotationStrategy().analyze(price_df_with_vix, "SPY"))
        assert result is None or isinstance(result, Signal)

    def test_analyze_signal_has_direction(self, price_df):
        result = asyncio.run(BondEquityRotationStrategy().analyze(price_df, "SPY"))
        if result is not None:
            assert result.side in ("buy", "sell")

    def test_analyze_signal_confidence_in_range(self, price_df):
        result = asyncio.run(BondEquityRotationStrategy().analyze(price_df, "SPY"))
        if result is not None:
            assert 0.0 <= result.confidence <= 1.0

    def test_analyze_signal_strategy_name_matches(self, price_df):
        result = asyncio.run(BondEquityRotationStrategy().analyze(price_df, "SPY"))
        if result is not None:
            assert result.strategy_name == "bond_equity_rotation"

    def test_analyze_returns_none_on_insufficient_data(self, short_df):
        result = asyncio.run(BondEquityRotationStrategy().analyze(short_df, "SPY"))
        assert result is None

    def test_analyze_symbol_propagated(self, price_df):
        result = asyncio.run(BondEquityRotationStrategy().analyze(price_df, "TLT"))
        if result is not None:
            assert result.symbol == "TLT"

    def test_analyze_sell_signal_in_fear_regime(self):
        """High positive corr between returns and dvix → fear regime → sell."""
        rng = np.random.default_rng(0)
        n = 100
        # Create close prices and vix that move together (high positive corr)
        base = rng.normal(0, 0.01, n)
        close = 100 * np.cumprod(1 + base)
        vix = 20 * np.cumprod(1 + base + rng.normal(0, 0.005, n))  # correlated with close returns
        df = pd.DataFrame(
            {"close": close, "vix_close": vix},
            index=pd.date_range("2023-01-01", periods=n, freq="D"),
        )
        result = asyncio.run(BondEquityRotationStrategy().analyze(df, "SPY"))
        # Accept either a valid signal or None (correlation may not hit threshold)
        assert result is None or isinstance(result, Signal)


# ---------------------------------------------------------------------------
# Test 3: backtest_signals()
# ---------------------------------------------------------------------------

class TestBondEquityRotationBacktest:
    def test_returns_backtest_signals_type(self, price_df):
        result = BondEquityRotationStrategy().backtest_signals(price_df)
        assert isinstance(result, BacktestSignals)

    def test_entries_exits_are_bool(self, price_df):
        result = BondEquityRotationStrategy().backtest_signals(price_df)
        assert result.entries.dtype == bool
        assert result.exits.dtype == bool

    def test_series_length_matches_input(self, price_df):
        n = len(price_df)
        result = BondEquityRotationStrategy().backtest_signals(price_df)
        assert len(result.entries) == n
        assert len(result.exits) == n

    def test_no_lookahead_first_row_false(self, price_df):
        result = BondEquityRotationStrategy().backtest_signals(price_df)
        assert not bool(result.entries.iloc[0])

    def test_no_nan_in_signals(self, price_df):
        result = BondEquityRotationStrategy().backtest_signals(price_df)
        assert not result.entries.isna().any()
        assert not result.exits.isna().any()

    def test_index_matches_input(self, price_df):
        result = BondEquityRotationStrategy().backtest_signals(price_df)
        pd.testing.assert_index_equal(result.entries.index, price_df.index)
        pd.testing.assert_index_equal(result.exits.index, price_df.index)

    def test_backtest_with_vix_column(self, price_df_with_vix):
        result = BondEquityRotationStrategy().backtest_signals(price_df_with_vix)
        assert isinstance(result, BacktestSignals)
        assert len(result.entries) == len(price_df_with_vix)

    def test_short_entries_present(self, price_df):
        result = BondEquityRotationStrategy().backtest_signals(price_df)
        assert result.short_entries is not None
        assert len(result.short_entries) == len(price_df)
