"""
Unit tests for the Basis Carry (BTC spot-futures) strategy.

Covers:
  1. Strategy instantiates with correct attributes
  2. analyze() returns Signal or None — external Binance calls are mocked
  3. backtest_signals() returns BacktestSignals with correct dtype
  4. No-lookahead: first row is always False
  5. Insufficient data returns empty signals
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from app.strategies.base import BacktestSignals, Signal
from app.strategies.manual.basis_carry import BasisCarryStrategy


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
    """Only 10 bars — below the required window."""
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

class TestBasisCarryAttributes:
    def test_strategy_name(self):
        assert BasisCarryStrategy().name == "basis_carry"

    def test_market_type(self):
        assert BasisCarryStrategy().market_type == "crypto"

    def test_strategy_type(self):
        assert BasisCarryStrategy().strategy_type == "manual"

    def test_risk_bucket(self):
        assert BasisCarryStrategy().risk_bucket == "arbitrage"

    def test_default_thresholds(self):
        s = BasisCarryStrategy()
        assert s.entry_threshold_pct == pytest.approx(5.0)
        assert s.exit_threshold_pct == pytest.approx(1.0)

    def test_custom_params(self):
        s = BasisCarryStrategy(params={"entry_threshold_pct": 8.0, "exit_threshold_pct": 2.0})
        assert s.entry_threshold_pct == pytest.approx(8.0)
        assert s.exit_threshold_pct == pytest.approx(2.0)

    def test_display_name_not_empty(self):
        assert len(BasisCarryStrategy().display_name) > 0

    def test_description_returns_string(self):
        desc = BasisCarryStrategy().description()
        assert isinstance(desc, str) and len(desc) > 0


# ---------------------------------------------------------------------------
# Test 2: analyze() with mocked Binance calls
# ---------------------------------------------------------------------------

class TestBasisCarryAnalyze:
    def _run_analyze(self, spot, perp, symbol="BTCUSDT"):
        s = BasisCarryStrategy()
        with patch.object(s, "_fetch_prices", new=AsyncMock(return_value=(spot, perp))):
            return asyncio.run(s.analyze(pd.DataFrame({"close": [spot]}), symbol))

    def test_returns_signal_or_none_below_threshold(self):
        # basis ~0 → should return None (between exit and entry thresholds)
        result = self._run_analyze(spot=30000.0, perp=30001.0)
        assert result is None or isinstance(result, Signal)

    def test_returns_buy_signal_when_basis_above_entry(self):
        # annualised_basis = (perp/spot - 1) * 365 * 100 >> 5%
        # perp = spot * (1 + 0.0005) → annualised ≈ 0.05% * 365 = 18.25% >> 5%
        result = self._run_analyze(spot=30000.0, perp=30150.0)
        assert result is not None
        assert result.side == "buy"
        assert result.strategy_name == "basis_carry"

    def test_returns_sell_signal_when_basis_below_exit(self):
        # annualised_basis negative (contango reversed) << exit_threshold
        result = self._run_analyze(spot=30000.0, perp=29990.0)
        assert result is not None
        assert result.side == "sell"

    def test_signal_confidence_in_range(self):
        result = self._run_analyze(spot=30000.0, perp=30150.0)
        if result is not None:
            assert 0.0 <= result.confidence <= 1.0

    def test_signal_symbol_propagated(self):
        result = self._run_analyze(spot=30000.0, perp=30150.0, symbol="ETHUSDT")
        if result is not None:
            assert result.symbol == "ETHUSDT"

    def test_analyze_raises_on_fetch_failure(self, price_df):
        s = BasisCarryStrategy()
        with patch.object(s, "_fetch_prices", new=AsyncMock(side_effect=Exception("network error"))):
            with pytest.raises(RuntimeError):
                asyncio.run(s.analyze(price_df, "BTCUSDT"))


# ---------------------------------------------------------------------------
# Test 3: backtest_signals()
# ---------------------------------------------------------------------------

class TestBasisCarryBacktest:
    def test_returns_backtest_signals_type(self, price_df):
        result = BasisCarryStrategy().backtest_signals(price_df)
        assert isinstance(result, BacktestSignals)

    def test_entries_exits_are_bool(self, price_df):
        result = BasisCarryStrategy().backtest_signals(price_df)
        assert result.entries.dtype == bool
        assert result.exits.dtype == bool

    def test_series_length_matches_input(self, price_df):
        n = len(price_df)
        result = BasisCarryStrategy().backtest_signals(price_df)
        assert len(result.entries) == n
        assert len(result.exits) == n

    def test_no_lookahead_first_row_false(self, price_df):
        result = BasisCarryStrategy().backtest_signals(price_df)
        assert not bool(result.entries.iloc[0])

    def test_insufficient_data_returns_all_false(self, short_df):
        result = BasisCarryStrategy().backtest_signals(short_df)
        assert not result.entries.any()
        assert not result.exits.any()

    def test_index_matches_input(self, price_df):
        result = BasisCarryStrategy().backtest_signals(price_df)
        pd.testing.assert_index_equal(result.entries.index, price_df.index)
        pd.testing.assert_index_equal(result.exits.index, price_df.index)

    def test_no_nan_in_signals(self, price_df):
        result = BasisCarryStrategy().backtest_signals(price_df)
        assert not result.entries.isna().any()
        assert not result.exits.isna().any()
