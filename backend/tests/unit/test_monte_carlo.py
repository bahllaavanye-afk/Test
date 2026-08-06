"""Monte Carlo simulation tests."""
import pandas as pd
import numpy as np
import pytest
from app.backtest.monte_carlo import monte_carlo_simulation


def test_monte_carlo_basic():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.015, 500))
    result = monte_carlo_simulation(returns, n_simulations=200, n_years=2)
    assert result.num_simulations == 200
    assert result.median_sharpe is not None
    assert 0 <= result.prob_positive_return <= 1


def test_monte_carlo_confidence_intervals():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.015, 500))
    result = monte_carlo_simulation(returns, n_simulations=200, n_years=2)
    assert result.p5_sharpe <= result.median_sharpe <= result.p95_sharpe


def test_monte_carlo_empty_returns():
    """An empty return series should raise a ValueError."""
    empty_returns = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        monte_carlo_simulation(empty_returns, n_simulations=10, n_years=1)


def test_monte_carlo_zero_simulations():
    """Zero simulations is invalid and should raise a ValueError."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.015, 100))
    with pytest.raises(ValueError):
        monte_carlo_simulation(returns, n_simulations=0, n_years=1)


def test_monte_carlo_one_year_boundary():
    """Simulate a single year to ensure probabilities stay within bounds."""
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.015, 250))
    result = monte_carlo_simulation(returns, n_simulations=50, n_years=1)
    assert result.num_simulations == 50
    assert 0 <= result.prob_positive_return <= 1
    # Median Sharpe should be finite and comparable to confidence bounds
    assert result.p5_sharpe <= result.median_sharpe <= result.p95_sharpe