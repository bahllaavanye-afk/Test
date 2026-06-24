"""Tests for VaR, factor exposure, and drawdown recovery."""
import numpy as np
import pytest
from app.risk.var import historical_var, VaRResult
from app.risk.factor_exposure import compute_factor_exposure, FactorExposure
from app.risk.drawdown_recovery import estimate_recovery

# Constants extracted from magic numbers / hardcoded strings
DEFAULT_PORTFOLIO_VALUE = 100_000
DEFAULT_VAR_95 = 0.02
DEFAULT_BETA = 1.0
BETA_LOWER_BOUND = 0.3
BETA_UPPER_BOUND = 1.3
METHOD_PARAMETRIC = "parametric"

VAR_DICT_KEYS = ("var_95_pct", "cvar_95_pct", "interpretation")
FACTOR_DICT_KEYS = ("market_beta", "alpha_annualized_pct", "interpretation")
DRAWDOWN_DICT_KEYS = ("expected_recovery_days", "probability_recover_30d")


class TestHistoricalVaR:
    def test_basic_output(self):
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.015, 252))
        result = historical_var(returns, portfolio_value=DEFAULT_PORTFOLIO_VALUE)
        assert isinstance(result, VaRResult)
        assert result.var_95 > 0
        assert result.var_99 >= result.var_95
        assert result.cvar_95 >= result.var_95  # CVaR >= VaR always

    def test_var_99_stricter_than_95(self):
        np.random.seed(1)
        returns = list(np.random.normal(0.0005, 0.02, 500))
        result = historical_var(returns, DEFAULT_PORTFOLIO_VALUE)
        assert result.var_99 >= result.var_95

    def test_insufficient_data_returns_defaults(self):
        result = historical_var([0.01, -0.02], DEFAULT_PORTFOLIO_VALUE)
        assert result.var_95 == DEFAULT_VAR_95  # default

    def test_to_dict_has_required_keys(self):
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.01, 100))
        d = historical_var(returns, 50_000).to_dict()
        for key in VAR_DICT_KEYS:
            assert key in d

    def test_parametric_method(self):
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.015, 252))
        result = historical_var(returns, DEFAULT_PORTFOLIO_VALUE, method=METHOD_PARAMETRIC)
        assert result.method == METHOD_PARAMETRIC
        assert result.var_95 > 0


class TestFactorExposure:
    def test_basic_computation(self):
        np.random.seed(42)
        market = list(np.random.normal(0.0004, 0.012, 252))
        portfolio = [m * 0.8 + np.random.normal(0, 0.005) for m in market]
        result = compute_factor_exposure(portfolio, market)
        assert isinstance(result, FactorExposure)
        # With β=0.8 market, should be close to 0.8
        assert BETA_LOWER_BOUND < result.market_beta < BETA_UPPER_BOUND

    def test_short_series(self):
        result = compute_factor_exposure([0.01, -0.02], [0.01, -0.02])
        assert result.market_beta == DEFAULT_BETA  # default

    def test_to_dict_keys(self):
        np.random.seed(0)
        r = compute_factor_exposure(
            list(np.random.normal(0, 0.01, 60)),
            list(np.random.normal(0, 0.01, 60)),
        )
        d = r.to_dict()
        for key in FACTOR_DICT_KEYS:
            assert key in d


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
        for key in DRAWDOWN_DICT_KEYS:
            assert key in d