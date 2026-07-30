"""Tests for VaR, factor exposure, and drawdown recovery."""
import numpy as np
import pytest
from app.risk.var import historical_var, VaRResult
from app.risk.factor_exposure import compute_factor_exposure, FactorExposure
from app.risk.drawdown_recovery import estimate_recovery


class TestHistoricalVaR:
    def test_basic_output(self):
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.015, 252))
        result = historical_var(returns, portfolio_value=100_000)
        assert isinstance(result, VaRResult)
        assert result.var_95 > 0
        assert result.var_99 >= result.var_95
        assert result.cvar_95 >= result.var_95  # CVaR >= VaR always
        assert result.cvar_99 >= result.var_99

    def test_var_99_stricter_than_95(self):
        np.random.seed(1)
        returns = list(np.random.normal(0.0005, 0.02, 500))
        result = historical_var(returns, 100_000)
        assert result.var_99 >= result.var_95
        # Ensure monotonicity between confidence levels
        assert result.cvar_99 >= result.cvar_95

    def test_insufficient_data_returns_defaults(self):
        result = historical_var([0.01, -0.02], 100_000)
        # Default VaR for insufficient data is the max absolute return
        assert result.var_95 == 0.02

    def test_to_dict_has_required_keys(self):
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.01, 100))
        d = historical_var(returns, 50_000).to_dict()
        for key in ("var_95_pct", "cvar_95_pct", "interpretation"):
            assert key in d

    def test_parametric_method_consistency(self):
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.015, 252))
        hist = historical_var(returns, 100_000, method="historical")
        para = historical_var(returns, 100_000, method="parametric")
        assert hist.method == "historical"
        assert para.method == "parametric"
        # Parametric VaR should be close but not lower than historical VaR for normal data
        assert para.var_95 >= hist.var_95 * 0.9

    def test_negative_returns_produce_positive_var(self):
        returns = [-0.03, -0.015, -0.02, -0.01]
        result = historical_var(returns, 50_000)
        assert result.var_95 > 0
        assert result.cvar_95 > result.var_95


class TestFactorExposure:
    def test_basic_computation(self):
        np.random.seed(42)
        market = list(np.random.normal(0.0004, 0.012, 252))
        portfolio = [m * 0.8 + np.random.normal(0, 0.005) for m in market]
        result = compute_factor_exposure(portfolio, market)
        assert isinstance(result, FactorExposure)
        # Tightened acceptance range for known beta of 0.8
        assert 0.7 < result.market_beta < 0.9

    def test_short_series_defaults(self):
        result = compute_factor_exposure([0.01, -0.02], [0.01, -0.02])
        # When insufficient data, fallback to neutral beta
        assert result.market_beta == 1.0

    def test_to_dict_keys(self):
        np.random.seed(0)
        r = compute_factor_exposure(
            list(np.random.normal(0, 0.01, 60)),
            list(np.random.normal(0, 0.01, 60)),
        )
        d = r.to_dict()
        for key in ("market_beta", "alpha_annualized_pct", "interpretation"):
            assert key in d

    def test_alpha_significance_filter(self):
        np.random.seed(123)
        market = list(np.random.normal(0, 0.01, 200))
        # Create a portfolio with a small drift relative to market
        portfolio = [m * 0.5 + 0.0002 for m in market]
        result = compute_factor_exposure(portfolio, market)
        # Alpha should be small; enforce a magnitude filter as confirmation
        assert abs(result.alpha_annualized_pct) < 0.05


class TestDrawdownRecovery:
    def test_no_drawdown(self):
        result = estimate_recovery([0.001] * 100, 0.0)
        assert result.current_drawdown_pct == 0

    def test_positive_drift_recovers(self):
        np.random.seed(42)
        returns = list(np.random.normal(0.002, 0.01, 252))  # strong positive drift
        result = estimate_recovery(returns, 0.05)
        assert result.expected_recovery_days is not None
        assert isinstance(result.expected_recovery_days, int)
        assert result.probability_recover_90d > 0.5

    def test_negative_drift_no_recovery(self):
        returns = [-0.002] * 100  # consistent losses
        result = estimate_recovery(returns, 0.10)
        assert result.expected_recovery_days is None

    def test_to_dict_structure(self):
        np.random.seed(1)
        returns = list(np.random.normal(0.001, 0.012, 252))
        d = estimate_recovery(returns, 0.03).to_dict()
        for key in ("expected_recovery_days", "probability_recover_30d"):
            assert key in d

    def test_probability_monotonicity(self):
        np.random.seed(7)
        returns = list(np.random.normal(0.0015, 0.009, 252))
        result = estimate_recovery(returns, 0.04)
        # Probability to recover should increase with longer horizon
        assert result.probability_recover_30d <= result.probability_recover_60d <= result.probability_recover_90d