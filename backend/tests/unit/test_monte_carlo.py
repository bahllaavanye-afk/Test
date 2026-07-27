"""Monte Carlo simulation tests."""
import pandas as pd
import numpy as np
from app.backtest.monte_carlo import monte_carlo_simulation


def _validate_inputs(returns, n_simulations, n_years):
    """Validate inputs for monte_carlo_simulation.

    Raises:
        ValueError: If any input is invalid.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series.")
    if returns.empty:
        raise ValueError("returns Series cannot be empty.")
    if not np.issubdtype(returns.dtype, np.floating):
        raise ValueError("returns Series must contain numeric (float) values.")
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    if not isinstance(n_years, (int, float)) or n_years <= 0:
        raise ValueError("n_years must be a positive number.")


def test_monte_carlo_basic():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.015, 500))
    _validate_inputs(returns, n_simulations=200, n_years=2)
    result = monte_carlo_simulation(returns, n_simulations=200, n_years=2)
    assert result.num_simulations == 200
    assert result.median_sharpe is not None
    assert 0 <= result.prob_positive_return <= 1


def test_monte_carlo_confidence_intervals():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.015, 500))
    _validate_inputs(returns, n_simulations=200, n_years=2)
    result = monte_carlo_simulation(returns, n_simulations=200, n_years=2)
    assert result.p5_sharpe <= result.median_sharpe <= result.p95_sharpe