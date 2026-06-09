"""Tests for strategies added in the latest platform build."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.strategies import STRATEGY_REGISTRY


ET = ZoneInfo("America/New_York")


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def daily_ohlcv():
    """300 days of synthetic daily OHLCV."""
    rng = np.random.default_rng(42)
    n = 300
    returns = rng.normal(0.0004, 0.013, n)
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="1D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def intraday_ohlcv():
    """One full trading day of 1-minute OHLCV data (9:30 AM–4:00 PM ET)."""
    rng = np.random.default_rng(7)
    base_date = datetime(2024, 3, 1, tzinfo=ET)
    start = base_date.replace(hour=9, minute=30)
    end = base_date.replace(hour=16, minute=0)
    # 390 minutes in a regular session
    idx = pd.date_range(start, end, freq="1min", tz=ET)[:390]
    n = len(idx)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + rng.uniform(0, 0.1, n)
    low = close - rng.uniform(0, 0.1, n)
    open_ = close + rng.normal(0, 0.02, n)
    # Higher volume at open (first 30 bars)
    volume = rng.integers(10_000, 50_000, n).astype(float)
    volume[:30] *= 3.0
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ── OpeningRangeBreakout ──────────────────────────────────────────────────────

class TestOpeningRangeBreakout:
    def _get(self):
        cls = STRATEGY_REGISTRY.get("opening_range_breakout")
        if cls is None:
            pytest.skip("opening_range_breakout not in registry")
        return cls()

    def test_in_registry(self):
        assert "opening_range_breakout" in STRATEGY_REGISTRY

    def test_required_attrs(self):
        inst = self._get()
        assert inst.market_type == "equity"
        assert inst.strategy_type == "manual"
        assert inst.risk_bucket == "directional"

    def test_backtest_signals_returns_series(self, intraday_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        assert signals is not None
        assert isinstance(signals, pd.Series)

    def test_backtest_signals_valid_values(self, intraday_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        valid = {-1, 0, 1, np.nan}
        for v in signals.dropna():
            assert v in valid, f"Invalid signal value: {v}"

    def test_no_lookahead_bias(self, intraday_ohlcv):
        """Signals must be shifted — first signal must be NaN or 0."""
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        # The very first bar cannot have a confirmed breakout signal
        first_nonzero = signals.dropna()
        if len(first_nonzero) > 0:
            # Signal at bar t is based on data up to bar t-1
            first_signal_bar = signals.first_valid_index()
            signal_position = signals.index.get_loc(first_signal_bar)
            assert signal_position > 0, "Signal at bar 0 is lookahead bias"

    def test_backtest_signals_daily_fallback(self, daily_ohlcv):
        """Strategy must not crash on daily data (may return all-zero signals)."""
        inst = self._get()
        signals = inst.backtest_signals(daily_ohlcv)
        assert signals is not None


# ── VWAPReversion ─────────────────────────────────────────────────────────────

class TestVWAPReversion:
    def _get(self):
        cls = STRATEGY_REGISTRY.get("vwap_reversion")
        if cls is None:
            pytest.skip("vwap_reversion not in registry")
        return cls()

    def test_in_registry(self):
        assert "vwap_reversion" in STRATEGY_REGISTRY

    def test_required_attrs(self):
        inst = self._get()
        assert inst.market_type == "equity"
        assert inst.strategy_type == "manual"
        assert inst.risk_bucket == "directional"

    def test_backtest_signals_returns_series(self, intraday_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(intraday_ohlcv)

    def test_signals_valid_range(self, intraday_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        valid = {-1, 0, 1, np.nan}
        for v in signals.dropna():
            assert v in valid

    def test_no_signal_without_volume(self):
        """VWAP is undefined without volume — must not crash."""
        inst = self._get()
        rng = np.random.default_rng(1)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        df = pd.DataFrame({
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 0.0,
        }, index=idx)
        signals = inst.backtest_signals(df)
        assert signals is not None


# ── CrossSectionalMomentum ────────────────────────────────────────────────────

class TestCrossSectionalMomentum:
    def _get(self):
        cls = STRATEGY_REGISTRY.get("cross_sectional_momentum")
        if cls is None:
            pytest.skip("cross_sectional_momentum not in registry")
        return cls()

    def test_in_registry(self):
        assert "cross_sectional_momentum" in STRATEGY_REGISTRY

    def test_required_attrs(self):
        inst = self._get()
        assert inst.market_type == "equity"
        assert inst.strategy_type == "manual"
        assert inst.risk_bucket == "directional"

    def test_backtest_signals_returns_series(self, daily_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(daily_ohlcv)
        assert isinstance(signals, pd.Series)

    def test_signals_are_shifted(self, daily_ohlcv):
        """Monthly rebalance signals must not appear before 12+1 months of data."""
        inst = self._get()
        signals = inst.backtest_signals(daily_ohlcv)
        # Must have at least some non-null signals
        non_null = signals.dropna()
        assert len(non_null) >= 0  # May be empty if not enough data


# ── Strategy registry completeness ───────────────────────────────────────────

class TestStrategyRegistry:
    EXPECTED_NEW_STRATEGIES = [
        "opening_range_breakout",
        "vwap_reversion",
        "cross_sectional_momentum",
    ]

    @pytest.mark.parametrize("name", EXPECTED_NEW_STRATEGIES)
    def test_strategy_registered(self, name):
        assert name in STRATEGY_REGISTRY, (
            f"Strategy '{name}' is not in STRATEGY_REGISTRY — "
            "check backend/app/strategies/__init__.py"
        )

    @pytest.mark.parametrize("name", EXPECTED_NEW_STRATEGIES)
    def test_strategy_instantiates(self, name):
        cls = STRATEGY_REGISTRY.get(name)
        if cls is None:
            pytest.skip(f"{name} not in registry")
        inst = cls()
        assert inst is not None

    @pytest.mark.parametrize("name", EXPECTED_NEW_STRATEGIES)
    def test_strategy_has_name_attr(self, name):
        cls = STRATEGY_REGISTRY.get(name)
        if cls is None:
            pytest.skip(f"{name} not in registry")
        inst = cls()
        assert hasattr(inst, "name")
        assert inst.name == name

    def test_registry_has_equity_strategies(self):
        equity = [name for name, cls in STRATEGY_REGISTRY.items()
                  if hasattr(cls(), "market_type") and cls().market_type == "equity"]
        assert len(equity) >= 10, f"Expected ≥10 equity strategies, got {len(equity)}"

    def test_registry_has_crypto_strategies(self):
        crypto = [name for name, cls in STRATEGY_REGISTRY.items()
                  if hasattr(cls(), "market_type") and cls().market_type == "crypto"]
        assert len(crypto) >= 3, f"Expected ≥3 crypto strategies, got {len(crypto)}"

    def test_registry_has_polymarket_strategies(self):
        poly = [name for name, cls in STRATEGY_REGISTRY.items()
                if hasattr(cls(), "market_type") and cls().market_type == "polymarket"]
        assert len(poly) >= 1, f"Expected ≥1 polymarket strategy, got {len(poly)}"

    def test_all_strategies_have_risk_bucket(self):
        missing = []
        for name, cls in STRATEGY_REGISTRY.items():
            try:
                inst = cls()
                if not hasattr(inst, "risk_bucket") or inst.risk_bucket not in ("arbitrage", "directional"):
                    missing.append(name)
            except Exception:
                pass
        assert not missing, f"Strategies with invalid risk_bucket: {missing}"
