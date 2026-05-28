"""
Unit tests for PCA Statistical Arbitrage strategies.

Tests cover:
  1. PCAStatArbStrategy instantiates correctly
  2. backtest_signals() returns BacktestSignals with correct dtypes
  3. Signals use shift(1) — first row is NaN / False before any data
  4. PCA basket can be configured via params
  5. MLPCAStatArbStrategy falls back gracefully when no ML model is loaded
"""
import asyncio

import numpy as np
import pandas as pd
import pytest

from app.strategies.base import BacktestSignals
from app.strategies.manual.pca_stat_arb import DEFAULT_BASKET, PCAStatArbStrategy
from app.strategies.ml_enhanced.ml_pca_arb import MLPCAStatArbStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_basket_df(
    n: int = 120,
    symbols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a flat DataFrame with  close_<SYMBOL>  columns for every symbol in
    *symbols* (defaults to the first 5 names from DEFAULT_BASKET).
    """
    if symbols is None:
        symbols = DEFAULT_BASKET[:5]

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    data: dict[str, np.ndarray] = {}
    for sym in symbols:
        returns = rng.normal(5e-4, 0.015, n)
        data[f"close_{sym}"] = 100.0 * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=idx)


def _make_multiindex_df(
    n: int = 120,
    symbols: list[str] | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Build a DataFrame with MultiIndex columns (symbol, field) containing
    'close', 'open', 'volume' for each symbol.
    """
    if symbols is None:
        symbols = DEFAULT_BASKET[:4]

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for sym in symbols:
        close = 100.0 * np.cumprod(1 + rng.normal(5e-4, 0.015, n))
        arrays[(sym, "close")] = close
        arrays[(sym, "open")] = close * (1 + rng.normal(0, 0.001, n))
        arrays[(sym, "volume")] = rng.integers(100_000, 1_000_000, n).astype(float)
    df = pd.DataFrame(arrays, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ---------------------------------------------------------------------------
# Test 1: instantiation
# ---------------------------------------------------------------------------

class TestPCAStatArbInstantiation:
    def test_default_instantiation(self):
        """Strategy should instantiate with sensible defaults."""
        s = PCAStatArbStrategy()
        assert s.name == "pca_stat_arb"
        assert s.strategy_type == "manual"
        assert s.risk_bucket == "arbitrage"
        assert s.basket == DEFAULT_BASKET
        assert s.n_components == 5
        assert s.lookback == 60
        assert s.entry_z == pytest.approx(1.5)
        assert s.exit_z == pytest.approx(0.5)
        assert s.stop_z == pytest.approx(3.5)

    def test_custom_params(self):
        """Params dict should override all defaults."""
        custom_basket = ["AAPL", "MSFT", "GOOGL"]
        s = PCAStatArbStrategy(params={
            "basket": custom_basket,
            "n_components": 2,
            "lookback": 30,
            "entry_z": 2.0,
            "exit_z": 1.0,
            "stop_z": 4.0,
        })
        assert s.basket == custom_basket
        assert s.n_components == 2
        assert s.lookback == 30
        assert s.entry_z == pytest.approx(2.0)

    def test_display_name_not_empty(self):
        s = PCAStatArbStrategy()
        assert len(s.display_name) > 0

    def test_description_method(self):
        s = PCAStatArbStrategy()
        desc = s.description()
        assert "manual" in desc.lower() or "pca" in desc.lower()


# ---------------------------------------------------------------------------
# Test 2: backtest_signals returns BacktestSignals with correct dtype
# ---------------------------------------------------------------------------

class TestBacktestSignalsReturn:
    def test_returns_backtest_signals_type(self):
        s = PCAStatArbStrategy()
        df = _make_basket_df(n=120)
        result = s.backtest_signals(df)
        assert isinstance(result, BacktestSignals)

    def test_entries_exits_are_bool_series(self):
        s = PCAStatArbStrategy()
        df = _make_basket_df(n=120)
        result = s.backtest_signals(df)
        assert result.entries.dtype == bool, f"entries dtype: {result.entries.dtype}"
        assert result.exits.dtype == bool, f"exits dtype: {result.exits.dtype}"

    def test_series_have_same_length_as_input(self):
        n = 120
        s = PCAStatArbStrategy()
        df = _make_basket_df(n=n)
        result = s.backtest_signals(df)
        assert len(result.entries) == n
        assert len(result.exits) == n

    def test_short_entries_and_exits_present(self):
        s = PCAStatArbStrategy()
        df = _make_basket_df(n=120)
        result = s.backtest_signals(df)
        assert result.short_entries is not None
        assert result.short_exits is not None

    def test_multiindex_columns_accepted(self):
        """Strategy should work with (symbol, field) MultiIndex columns."""
        symbols = DEFAULT_BASKET[:5]
        s = PCAStatArbStrategy(params={"basket": symbols})
        df = _make_multiindex_df(n=120, symbols=symbols)
        result = s.backtest_signals(df)
        assert isinstance(result, BacktestSignals)
        assert len(result.entries) == 120

    def test_empty_df_returns_false_signals(self):
        """When no basket columns present, should return all-False signals."""
        s = PCAStatArbStrategy()
        df = pd.DataFrame({"irrelevant_col": [1, 2, 3]})
        result = s.backtest_signals(df)
        assert not result.entries.any()
        assert not result.exits.any()


# ---------------------------------------------------------------------------
# Test 3: shift(1) — no lookahead bias
# ---------------------------------------------------------------------------

class TestNoLookaheadBias:
    def test_first_row_is_false_entries(self):
        """
        After shift(1) the very first row of entries must be False/NaN —
        no data has been seen yet at bar 0.
        """
        s = PCAStatArbStrategy(params={"lookback": 30})
        df = _make_basket_df(n=120)
        result = s.backtest_signals(df)
        # First row should never be True (shift pushes everything by 1)
        assert not bool(result.entries.iloc[0]), "entries[0] must be False after shift(1)"

    def test_first_row_is_false_short_entries(self):
        s = PCAStatArbStrategy(params={"lookback": 30})
        df = _make_basket_df(n=120)
        result = s.backtest_signals(df)
        assert not bool(result.short_entries.iloc[0]), "short_entries[0] must be False after shift(1)"

    def test_signals_index_matches_input_index(self):
        """Signal index must exactly match the input DataFrame index."""
        s = PCAStatArbStrategy()
        df = _make_basket_df(n=100)
        result = s.backtest_signals(df)
        pd.testing.assert_index_equal(result.entries.index, df.index)
        pd.testing.assert_index_equal(result.exits.index, df.index)


# ---------------------------------------------------------------------------
# Test 4: basket configuration
# ---------------------------------------------------------------------------

class TestBasketConfiguration:
    def test_custom_small_basket(self):
        """A 3-symbol basket is valid and should produce coherent output."""
        basket = ["AAPL", "MSFT", "GOOGL"]
        s = PCAStatArbStrategy(params={"basket": basket, "n_components": 2, "lookback": 30})
        df = _make_basket_df(n=80, symbols=basket)
        result = s.backtest_signals(df)
        assert isinstance(result, BacktestSignals)
        assert len(result.entries) == 80

    def test_basket_missing_columns_returns_empty(self):
        """
        When the DataFrame contains none of the basket symbols, backtest_signals
        must return all-False signals — not raise an exception.
        """
        basket = ["AAPL", "MSFT"]
        s = PCAStatArbStrategy(params={"basket": basket})
        df = pd.DataFrame({"close_XYZ": [1.0, 2.0, 3.0]})
        result = s.backtest_signals(df)
        assert not result.entries.any()

    def test_basket_partial_overlap(self):
        """
        If only some basket symbols are present the strategy should still work
        using the available subset (≥ 2 symbols needed for PCA).
        """
        basket = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        s = PCAStatArbStrategy(params={"basket": basket, "n_components": 2, "lookback": 30})
        # Only provide 2 of the 4 basket symbols
        df = _make_basket_df(n=80, symbols=["AAPL", "MSFT"])
        result = s.backtest_signals(df)
        assert isinstance(result, BacktestSignals)

    def test_different_basket_in_registry(self):
        """PCAStatArbStrategy should be importable from the strategy registry."""
        from app.strategies import STRATEGY_REGISTRY
        assert "pca_stat_arb" in STRATEGY_REGISTRY
        cls = STRATEGY_REGISTRY["pca_stat_arb"]
        inst = cls(params={"basket": DEFAULT_BASKET[:3]})
        assert inst.basket == DEFAULT_BASKET[:3]


# ---------------------------------------------------------------------------
# Test 5: ML version fallback when no model loaded
# ---------------------------------------------------------------------------

class TestMLPCAStatArbFallback:
    def test_ml_strategy_instantiates(self):
        """MLPCAStatArbStrategy should instantiate without errors."""
        s = MLPCAStatArbStrategy()
        assert s.name == "ml_pca_arb"
        assert s.strategy_type == "ml_enhanced"
        assert s.risk_bucket == "arbitrage"

    def test_ml_strategy_in_registry(self):
        from app.strategies import STRATEGY_REGISTRY
        assert "ml_pca_arb" in STRATEGY_REGISTRY

    def test_ml_analyze_returns_none_without_model(self):
        """
        When the ML inference service is unavailable (typical unit-test env),
        analyze() should return None without raising an exception.
        """
        s = MLPCAStatArbStrategy()
        df = _make_basket_df(n=120)

        async def _run():
            return await s.analyze(df, DEFAULT_BASKET[0])

        result = asyncio.run(_run())
        # Without a live ML service the result must be None (graceful fallback)
        assert result is None

    def test_ml_backtest_signals_delegates_to_base(self):
        """
        backtest_signals() on the ML version must return a valid BacktestSignals
        by delegating to the underlying PCA strategy — no ML service needed.
        """
        s = MLPCAStatArbStrategy()
        df = _make_basket_df(n=120)
        result = s.backtest_signals(df)
        assert isinstance(result, BacktestSignals)
        assert len(result.entries) == 120
        assert result.entries.dtype == bool

    def test_ml_backtest_no_lookahead(self):
        """ML version's delegated signals must also respect shift(1)."""
        s = MLPCAStatArbStrategy(params={"lookback": 30})
        df = _make_basket_df(n=120)
        result = s.backtest_signals(df)
        assert not bool(result.entries.iloc[0])

    def test_ml_custom_threshold_param(self):
        """ml_confidence_threshold param should be accepted."""
        s = MLPCAStatArbStrategy(params={"ml_confidence_threshold": 0.75})
        assert s._ml_threshold == pytest.approx(0.75)
