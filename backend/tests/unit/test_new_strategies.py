"""Tests for strategies added in the latest platform build."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field, validator

from app.strategies import STRATEGY_REGISTRY
from app.strategies.base import BacktestSignals


class OhlcvSchema(BaseModel):
    """Pydantic schema for OHLCV data used in strategy back‑testing.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame containing the OHLCV columns. Must include the columns
        ``open``, ``high``, ``low``, ``close`` and ``volume``. The index must be a
        ``DatetimeIndex`` with timezone information.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> from zoneinfo import ZoneInfo
    >>> idx = pd.date_range(datetime(2023, 1, 1, tzinfo=ZoneInfo("America/New_York")), periods=5, freq="1D")
    >>> df = pd.DataFrame(
    ...     {
    ...         "open": [100, 101, 102, 103, 104],
    ...         "high": [101, 102, 103, 104, 105],
    ...         "low": [99, 100, 101, 102, 103],
    ...         "close": [100.5, 101.5, 102.5, 103.5, 104.5],
    ...         "volume": [1_000_000, 1_200_000, 1_100_000, 1_300_000, 1_250_000],
    ...     },
    ...     index=idx,
    ... )
    >>> OhlcvSchema(df=df)  # doctest: +IGNORE_RESULT
    OhlcvSchema(df=...

    """

    df: pd.DataFrame = Field(
        ...,
        description="OHLCV DataFrame with required columns and a timezone‑aware DatetimeIndex.",
        example=pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1_000_000.0, 1_200_000.0],
            },
            index=pd.date_range(
                datetime(2023, 1, 1, tzinfo=ZoneInfo("America/New_York")),
                periods=2,
                freq="1D",
            ),
        ),
    )

    @validator("df")
    def validate_ohlcv(cls, value: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(value.columns)
        if missing:
            raise ValueError(f"OHLCV DataFrame missing required columns: {sorted(missing)}")
        if not isinstance(value.index, pd.DatetimeIndex):
            raise TypeError("OHLCV DataFrame index must be a pandas DatetimeIndex.")
        if value.index.tz is None:
            raise ValueError("OHLCV DataFrame index must be timezone‑aware.")
        if (value["volume"] < 0).any():
            raise ValueError("Volume values must be non‑negative.")
        return value

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


def _entries(sig) -> pd.Series:
    """Normalise both BacktestSignals and legacy pd.Series into an entries Series."""
    if isinstance(sig, BacktestSignals):
        return sig.entries
    # Legacy: float Series where 1 = long entry, -1 = short entry, 0 = flat
    return sig > 0


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
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    # Validate using the schema to ensure consistency across tests
    OhlcvSchema(df=df)
    return df


@pytest.fixture
def intraday_ohlcv():
    """One full trading day of 1‑minute OHLCV data (9:30 AM–4:00 PM ET)."""
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
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    OhlcvSchema(df=df)
    return df


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

    def test_backtest_signals_returns_backtestsignals(self, intraday_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        assert signals is not None
        assert isinstance(signals, BacktestSignals)
        assert isinstance(signals.entries, pd.Series)
        assert isinstance(signals.exits, pd.Series)

    def test_backtest_signals_bool_dtype(self, intraday_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        assert signals.entries.dtype == bool, "entries must be bool Series"
        assert signals.exits.dtype == bool, "exits must be bool Series"

    def test_no_lookahead_bias(self, intraday_ohlcv):
        """First bar cannot have an entry — range not yet established."""
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        assert not signals.entries.iloc[0], "Entry at bar 0 is lookahead bias"

    def test_backtest_signals_daily_fallback(self, daily_ohlcv):
        """Strategy must not crash on daily data (may return all‑False signals)."""
        inst = self._get()
        signals = inst.backtest_signals(daily_ohlcv)
        assert signals is not None
        assert isinstance(signals, BacktestSignals)


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

    def test_backtest_signals_returns_valid_type(self, intraday_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        assert isinstance(signals, (BacktestSignals, pd.Series))
        entries = _entries(signals)
        assert len(entries) == len(intraday_ohlcv)

    def test_signals_no_nan_in_entries(self, intraday_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(intraday_ohlcv)
        entries = _entries(signals)
        assert not entries.isna().any(), "entries must have no NaN"

    def test_no_signal_without_volume(self):
        """VWAP is undefined without volume — must not crash."""
        inst = self._get()
        rng = np.random.default_rng(1)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 0.0,
            },
            index=idx,
        )
        OhlcvSchema(df=df)
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

    def test_backtest_signals_returns_valid_type(self, daily_ohlcv):
        inst = self._get()
        signals = inst.backtest_signals(daily_ohlcv)
        assert isinstance(signals, (BacktestSignals, pd.Series))
        entries = _entries(signals)
        assert len(entries) == len(daily_ohlcv)

    def test_signals_are_shifted(self, daily_ohlcv):
        """Monthly rebalance signals must not appear before enough data."""
        inst = self._get()
        signals = inst.backtest_signals(daily_ohlcv)
        entries = _entries(signals)
        if entries.any():
            first_entry_pos = int(entries.values.argmax())
            assert first_entry_pos > 0, "Entry at bar 0 is lookahead bias"


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