"""Monte Carlo simulation tests."""
import pandas as pd
import numpy as np
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
