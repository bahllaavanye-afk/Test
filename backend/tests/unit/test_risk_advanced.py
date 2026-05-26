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

    def test_var_99_stricter_than_95(self):
        np.random.seed(1)
        returns = list(np.random.normal(0.0005, 0.02, 500))
        result = historical_var(returns, 100_000)
        assert result.var_99 >= result.var_95

    def test_insufficient_data_returns_defaults(self):
        result = historical_var([0.01, -0.02], 100_000)
        assert result.var_95 == 0.02  # default

    def test_to_dict_has_required_keys(self):
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.01, 100))
        d = historical_var(returns, 50_000).to_dict()
        assert "var_95_pct" in d
        assert "cvar_95_pct" in d
        assert "interpretation" in d

    def test_parametric_method(self):
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.015, 252))
        result = historical_var(returns, 100_000, method="parametric")
        assert result.method == "parametric"
        assert result.var_95 > 0


class TestFactorExposure:
    def test_basic_computation(self):
        np.random.seed(42)
        market = list(np.random.normal(0.0004, 0.012, 252))
        portfolio = [m * 0.8 + np.random.normal(0, 0.005) for m in market]
        result = compute_factor_exposure(portfolio, market)
        assert isinstance(result, FactorExposure)
        # With β=0.8 market, should be close to 0.8
        assert 0.3 < result.market_beta < 1.3

    def test_short_series(self):
        result = compute_factor_exposure([0.01, -0.02], [0.01, -0.02])
        assert result.market_beta == 1.0  # default

    def test_to_dict_keys(self):
        np.random.seed(0)
        r = compute_factor_exposure(list(np.random.normal(0, 0.01, 60)), list(np.random.normal(0, 0.01, 60)))
        d = r.to_dict()
        assert "market_beta" in d
        assert "alpha_annualized_pct" in d
        assert "interpretation" in d


class TestDrawdownRecovery:
    def test_no_drawdown(self):
        result = estimate_recovery([0.001] * 100, 0.0)
        assert result.current_drawdown_pct == 0

    def test_positive_drift_recovers(self):
        np.random.seed(42)
        returns = list(np.random.normal(0.002, 0.01, 252))  # strong positive drift
        result = estimate_recovery(returns, 0.05)
        assert result.expected_recovery_days is not None
        assert result.probability_recover_90d > 0.5

    def test_negative_drift_no_recovery(self):
        returns = [-0.002] * 100  # consistent losses
        result = estimate_recovery(returns, 0.10)
        assert result.expected_recovery_days is None

    def test_to_dict(self):
        np.random.seed(1)
        returns = list(np.random.normal(0.001, 0.012, 252))
        d = estimate_recovery(returns, 0.03).to_dict()
        assert "expected_recovery_days" in d
        assert "probability_recover_30d" in d
