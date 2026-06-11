"""Tests for HRP and CVaR portfolio optimizers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.risk.hrp import HRPOptimizer, _corr_to_distance, _get_quasi_diag
from app.risk.portfolio_optimizer import CVaROptimizer, optimize_portfolio


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def returns_df():
    """5 assets, 200 daily returns."""
    rng = np.random.default_rng(42)
    n, k = 200, 5
    # Introduce some correlation structure
    factor = rng.normal(0, 0.01, n)
    data = {}
    for i in range(k):
        beta = rng.uniform(0.3, 1.2)
        idio = rng.normal(0, 0.015, n)
        data[f"asset_{i}"] = beta * factor + idio
    return pd.DataFrame(data)


@pytest.fixture
def returns_df_large():
    """20 assets, 500 daily returns."""
    rng = np.random.default_rng(99)
    n, k = 500, 20
    factor = rng.normal(0, 0.01, n)
    data = {}
    for i in range(k):
        beta = rng.uniform(0.2, 1.5)
        idio = rng.normal(0, 0.012, n)
        data[f"A{i}"] = beta * factor + idio
    return pd.DataFrame(data)


# ── HRPOptimizer ──────────────────────────────────────────────────────────────

class TestHRPOptimizer:
    def test_weights_sum_to_one(self, returns_df):
        opt = HRPOptimizer()
        w = opt.compute_weights(returns_df)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_weights_all_positive(self, returns_df):
        opt = HRPOptimizer()
        w = opt.compute_weights(returns_df)
        assert (w >= 0).all()

    def test_weights_indexed_by_columns(self, returns_df):
        opt = HRPOptimizer()
        w = opt.compute_weights(returns_df)
        assert set(w.index) == set(returns_df.columns)

    def test_single_asset(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"A": rng.normal(0, 0.01, 100)})
        opt = HRPOptimizer()
        w = opt.compute_weights(df)
        assert abs(w["A"] - 1.0) < 1e-9

    def test_two_identical_assets_equal_weight(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0, 0.01, 100)
        df = pd.DataFrame({"A": r, "B": r})
        opt = HRPOptimizer()
        w = opt.compute_weights(df)
        assert abs(w["A"] - w["B"]) < 0.01

    def test_large_universe(self, returns_df_large):
        opt = HRPOptimizer()
        w = opt.compute_weights(returns_df_large)
        assert abs(w.sum() - 1.0) < 1e-9
        assert len(w) == 20

    def test_weights_stable_under_reorder(self, returns_df):
        opt = HRPOptimizer()
        w1 = opt.compute_weights(returns_df)
        w2 = opt.compute_weights(returns_df[returns_df.columns[::-1]])
        # Weights should be the same regardless of column order
        for col in returns_df.columns:
            assert abs(w1[col] - w2[col]) < 0.02, f"Weight for {col} differs by reordering"

    def test_high_corr_assets_get_lower_combined_weight(self):
        rng = np.random.default_rng(3)
        n = 300
        common = rng.normal(0, 0.01, n)
        df = pd.DataFrame({
            "correlated_1": common + rng.normal(0, 0.001, n),
            "correlated_2": common + rng.normal(0, 0.001, n),
            "independent":  rng.normal(0, 0.01, n),
        })
        opt = HRPOptimizer()
        w = opt.compute_weights(df)
        # The two correlated assets together should not dominate
        combined_corr = w["correlated_1"] + w["correlated_2"]
        assert combined_corr < 0.75, (
            f"HRP gave {combined_corr:.2f} to two highly correlated assets (expected < 0.75)"
        )


# ── HRP helper functions ──────────────────────────────────────────────────────

class TestHRPHelpers:
    def test_corr_to_distance_diagonal_zero(self):
        corr = pd.DataFrame(
            [[1.0, 0.5, 0.2],
             [0.5, 1.0, 0.3],
             [0.2, 0.3, 1.0]],
            columns=["A", "B", "C"],
        )
        dist = _corr_to_distance(corr)
        np.testing.assert_array_almost_equal(np.diag(dist), [0, 0, 0])

    def test_corr_to_distance_range(self):
        rng = np.random.default_rng(5)
        n = 10
        raw = rng.normal(0, 1, (300, n))
        corr = pd.DataFrame(np.corrcoef(raw.T))
        dist = _corr_to_distance(corr)
        assert (dist >= 0).all(), "Distances must be non-negative"
        assert (dist <= 1.0 + 1e-9).all(), "Distances must be <= 1"

    def test_perfect_correlation_zero_distance(self):
        corr = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]])
        dist = _corr_to_distance(corr)
        assert abs(dist[0, 1]) < 1e-9

    def test_negative_correlation_max_distance(self):
        corr = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]])
        dist = _corr_to_distance(corr)
        assert abs(dist[0, 1] - 1.0) < 1e-9


# ── CVaROptimizer ─────────────────────────────────────────────────────────────

class TestCVaROptimizer:
    def test_weights_sum_to_one(self, returns_df):
        opt = CVaROptimizer(confidence=0.95)
        w = opt.compute_weights(returns_df)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, returns_df):
        opt = CVaROptimizer(confidence=0.95)
        w = opt.compute_weights(returns_df)
        assert (w >= -1e-6).all()

    def test_indexed_by_columns(self, returns_df):
        opt = CVaROptimizer(confidence=0.95)
        w = opt.compute_weights(returns_df)
        assert set(w.index) == set(returns_df.columns)

    def test_confidence_validation(self):
        with pytest.raises((ValueError, AssertionError)):
            CVaROptimizer(confidence=0.0)
        with pytest.raises((ValueError, AssertionError)):
            CVaROptimizer(confidence=1.5)

    def test_different_confidence_levels(self, returns_df):
        w95 = CVaROptimizer(confidence=0.95).compute_weights(returns_df)
        w99 = CVaROptimizer(confidence=0.99).compute_weights(returns_df)
        assert abs(w95.sum() - 1.0) < 1e-6
        assert abs(w99.sum() - 1.0) < 1e-6

    def test_min_cvar_less_than_equal_weight(self, returns_df):
        opt = CVaROptimizer(confidence=0.95)
        w_opt = opt.compute_weights(returns_df)
        n = len(returns_df.columns)
        w_equal = pd.Series(1.0 / n, index=returns_df.columns)
        # Compute CVaR for each
        pnl_opt = (returns_df * w_opt).sum(axis=1).values
        pnl_eq = (returns_df * w_equal).sum(axis=1).values
        cutoff = int((1 - 0.95) * len(pnl_opt))
        cvar_opt = -np.sort(pnl_opt)[:cutoff].mean() if cutoff > 0 else 0
        cvar_eq = -np.sort(pnl_eq)[:cutoff].mean() if cutoff > 0 else 0
        assert cvar_opt <= cvar_eq + 0.005, (
            f"CVaR optimizer ({cvar_opt:.4f}) worse than equal weight ({cvar_eq:.4f})"
        )


# ── optimize_portfolio convenience function ───────────────────────────────────

class TestOptimizePortfolio:
    def test_hrp_method(self, returns_df):
        w = optimize_portfolio(returns_df, method="hrp")
        assert abs(w.sum() - 1.0) < 1e-9

    def test_cvar_method(self, returns_df):
        w = optimize_portfolio(returns_df, method="cvar")
        assert abs(w.sum() - 1.0) < 1e-6

    def test_equal_weight_fallback(self, returns_df):
        w = optimize_portfolio(returns_df, method="equal")
        n = len(returns_df.columns)
        for v in w.values:
            assert abs(v - 1.0 / n) < 1e-9

    def test_unknown_method_raises(self, returns_df):
        with pytest.raises((ValueError, KeyError, NotImplementedError)):
            optimize_portfolio(returns_df, method="bogus_method_xyz")
