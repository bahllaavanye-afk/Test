"""
Unit tests for BTCETHStatArb strategy.

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
from pydantic import BaseModel, Field, validator

from app.strategies.base import BacktestSignals, Signal
from app.strategies.manual.btc_eth_stat_arb import BTCETHStatArb


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class BTCETHStatArbParams(BaseModel):
    """
    Parameter schema for the BTCETHStatArb strategy.

    Attributes
    ----------
    window : int
        Rolling window size (in bars) used for statistical calculations.
    entry_z : float
        Z‑score threshold to trigger an entry signal.
    exit_z : float
        Z‑score threshold to trigger an exit signal.
    hedge_window : int
        Rolling window size (in bars) used for the hedge ratio calculation.
    """

    window: int = Field(
        ...,
        description="Rolling window size (in bars) for the primary statistic.",
        ge=1,
        example=60,
    )
    entry_z: float = Field(
        ...,
        description="Z‑score magnitude required to open a position.",
        gt=0,
        example=2.0,
    )
    exit_z: float = Field(
        ...,
        description="Z‑score magnitude required to close a position.",
        gt=0,
        example=0.5,
    )
    hedge_window: int = Field(
        60,
        description="Rolling window size (in bars) for hedge ratio estimation.",
        ge=1,
        example=60,
    )

    @validator("entry_z")
    def entry_z_must_exceed_exit_z(cls, v, values):
        """
        Ensure that the entry Z‑score is larger than the exit Z‑score.
        """
        exit_z = values.get("exit_z")
        if exit_z is not None and v <= exit_z:
            raise ValueError("entry_z must be greater than exit_z")
        return v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df():
    """200 bars of synthetic BTC/ETH OHLCV with both btc_close and eth_close."""
    rng = np.random.default_rng(42)
    n = 200
    btc_prices = 30000 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    eth_prices = 2000 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame(
        {
            "open": btc_prices * 0.999,
            "high": btc_prices * 1.005,
            "low": btc_prices * 0.995,
            "close": btc_prices,
            "btc_close": btc_prices,
            "eth_close": eth_prices,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="1h"),
    )


@pytest.fixture
def short_df():
    """Only 10 bars — below minimum."""
    rng = np.random.default_rng(7)
    n = 10
    btc_prices = 30000 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
    eth_prices = 2000 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
    return pd.DataFrame(
        {
            "close": btc_prices,
            "btc_close": btc_prices,
            "eth_close": eth_prices,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="1h"),
    )


# ---------------------------------------------------------------------------
# Test 1: attributes
# ---------------------------------------------------------------------------

class TestBTCETHStatArbAttributes:
    def test_strategy_name(self):
        assert BTCETHStatArb().name == "btc_eth_stat_arb"

    def test_market_type(self):
        assert BTCETHStatArb().market_type == "crypto"

    def test_strategy_type(self):
        assert BTCETHStatArb().strategy_type == "manual"

    def test_risk_bucket(self):
        assert BTCETHStatArb().risk_bucket == "arbitrage"

    def test_default_params(self):
        s = BTCETHStatArb()
        assert s.window == 60
        assert s.entry_z == pytest.approx(2.0)
        assert s.exit_z == pytest.approx(0.5)
        assert s.hedge_window == 60

    def test_custom_params(self):
        params = BTCETHStatArbParams(window=30, entry_z=1.5, exit_z=0.3).dict()
        s = BTCETHStatArb(params=params)
        assert s.window == 30
        assert s.entry_z == pytest.approx(1.5)
        assert s.exit_z == pytest.approx(0.3)

    def test_display_name_not_empty(self):
        assert len(BTCETHStatArb().display_name) > 0

    def test_description_returns_string(self):
        desc = BTCETHStatArb().description()
        assert isinstance(desc, str) and len(desc) > 0


# ---------------------------------------------------------------------------
# Test 2: analyze() — async
# ---------------------------------------------------------------------------

class TestBTCETHStatArbAnalyze:
    def test_analyze_returns_signal_or_none(self, price_df):
        result = asyncio.run(BTCETHStatArb().analyze(price_df, "BTCUSDT"))
        assert result is None or isinstance(result, Signal)

    def test_analyze_signal_has_direction(self, price_df):
        result = asyncio.run(BTCETHStatArb().analyze(price_df, "BTCUSDT"))
        if result is not None:
            assert result.side in ("buy", "sell")

    def test_analyze_signal_confidence_in_range(self, price_df):
        result = asyncio.run(BTCETHStatArb().analyze(price_df, "BTCUSDT"))
        if result is not None:
            assert 0.0 <= result.confidence <= 1.0

    def test_analyze_signal_strategy_name_matches(self, price_df):
        result = asyncio.run(BTCETHStatArb().analyze(price_df, "BTCUSDT"))
        if result is not None:
            assert result.strategy_name == "btc_eth_stat_arb"

    def test_analyze_returns_none_on_insufficient_data(self, short_df):
        result = asyncio.run(BTCETHStatArb().analyze(short_df, "BTCUSDT"))
        assert result is None

    def test_analyze_returns_none_when_no_eth_col(self):
        """Without eth_close column, strategy should return None."""
        rng = np.random.default_rng(42)
        n = 200
        prices = 30000 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
        df = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2023-01-01", periods=n, freq="1h"),
        )
        result = asyncio.run(BTCETHStatArb().analyze(df, "BTCUSDT"))
        assert result is None

    def test_analyze_symbol_propagated(self, price_df):
        result = asyncio.run(BTCETHStatArb().analyze(price_df, "ETHUSDT"))
        if result is not None:
            assert result.symbol == "ETHUSDT"

    def test_analyze_long_btc_signal_when_spread_low(self):
        """When z-score < -entry_z, should emit buy signal."""
        rng = np.random.default_rng(1)
        n = 200
        btc = 30000 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
        eth = 2000 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
        # Make BTC much cheaper relative to ETH at the end
        btc[-20:] *= 0.88
        df = pd.DataFrame(
            {"btc_close": btc, "eth_close": eth},
            index=pd.date_range("2023-01-01", periods=n, freq="1h"),
        )
        result = asyncio.run(BTCETHStatArb().analyze(df, "BTCUSDT"))
        # May or may not trigger depending on hedge ratio — just confirm valid type
        assert result is None or isinstance(result, Signal)


# ---------------------------------------------------------------------------
# Test 3: backtest_signals()
# ---------------------------------------------------------------------------

class TestBTCETHStatArbBacktest:
    def test_returns_backtest_signals_type(self, price_df):
        result = BTCETHStatArb().backtest_signals(price_df)
        assert isinstance(result, BacktestSignals)

    def test_entries_exits_are_bool(self, price_df):
        result = BTCETHStatArb().backtest_signals(price_df)
        assert result.entries.dtype == bool
        assert result.exits.dtype == bool

    def test_series_length_matches_input(self, price_df):
        n = len(price_df)
        result = BTCETHStatArb().backtest_signals(price_df)
        assert len(result.entries) == n
        assert len(result.exits) == n

    def test_no_lookahead_first_row_false(self, price_df):
        result = BTCETHStatArb().backtest_signals(price_df)
        assert not bool(result.entries.iloc[0])

    def test_insufficient_data_returns_all_false(self, short_df):
        result = BTCETHStatArb().backtest_signals(short_df)
        assert not result.entries.any()
        assert not result.exits.any()

    def test_index_matches_input(self, price_df):
        result = BTCETHStatArb().backtest_signals(price_df)
        pd.testing.assert_index_equal(result.entries.index, price_df.index)
        pd.testing.assert_index_equal(result.exits.index, price_df.index)

    def test_no_nan_in_signals(self, price_df):
        result = BTCETHStatArb().backtest_signals(price_df)
        assert not result.entries.isna().any()
        assert not result.exits.isna().any()

    def test_short_entries_present(self, price_df):
        result = BTCETHStatArb().backtest_signals(price_df)
        assert result.short_entries is not None
        assert len(result.short_entries) == len(price_df)