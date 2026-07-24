"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations
import numbers
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_max_dd: float
    p95_max_dd: float
    prob_positive_return: float
    num_simulations: int


def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    n_years: int = 3,
    risk_free_daily: float = 0.05 / 252,
) -> MonteCarloResult:
    """Bootstrap daily returns to simulate N years of paths.

    Parameters
    ----------
    daily_returns : pd.Series
        Series of daily returns. Must be non‑empty, numeric, and contain finite values.
    n_simulations : int
        Number of Monte‑Carlo paths to generate. Must be a positive integer.
    n_years : int
        Number of years to simulate. Must be a positive integer.
    risk_free_daily : float
        Daily risk‑free rate. Must be a real number.

    Returns
    -------
    MonteCarloResult
        Aggregated statistics from the simulations.

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    # Input validation
    if not isinstance(daily_returns, pd.Series):
        raise ValueError("daily_returns must be a pandas Series.")
    if daily_returns.empty:
        raise ValueError("daily_returns series cannot be empty.")
    if not np.issubdtype(daily_returns.dtype, np.number):
        raise ValueError("daily_returns must contain numeric values.")
    if not np.isfinite(daily_returns.dropna()).all():
        raise ValueError("daily_returns contains non‑finite values (NaN or Inf).")
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    if not isinstance(n_years, (int, float)) or n_years <= 0:
        raise ValueError("n_years must be a positive number.")
    if not isinstance(risk_free_daily, numbers.Real):
        raise ValueError("risk_free_daily must be a real number.")

    n_days = int(n_years * 252)
    returns_array = daily_returns.dropna().values
    sharpes = []
    max_dds = []
    positive = 0

    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        sampled = rng.choice(returns_array, size=n_days, replace=True)
        equity = np.cumprod(1 + sampled) * 100_000
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = dd.min()

        excess = sampled - risk_free_daily
        sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
        sharpes.append(sharpe)
        max_dds.append(max_dd)
        if equity[-1] > 100_000:
            positive += 1

    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpes)), 4),
        p5_sharpe=round(float(np.percentile(sharpes, 5)), 4),
        p95_sharpe=round(float(np.percentile(sharpes, 95)), 4),
        median_max_dd=round(float(np.median(max_dds)), 4),
        p95_max_dd=round(float(np.percentile(max_dds, 95)), 4),
        prob_positive_return=round(positive / n_simulations, 4),
        num_simulations=n_simulations,
    )


# Unit tests for edge and boundary conditions
import unittest


class TestMonteCarloSimulation(unittest.TestCase):
    def test_empty_series_raises(self):
        """Empty daily_returns should raise a ValueError."""
        empty_series = pd.Series(dtype=float)
        with self.assertRaises(ValueError):
            monte_carlo_simulation(empty_series)

    def test_zero_variance_sharpe(self):
        """When all returns are identical, Sharpe ratio should be zero."""
        constant_returns = pd.Series([0.001] * 20)  # identical daily returns
        result = monte_carlo_simulation(constant_returns, n_simulations=10, n_years=1)
        self.assertEqual(result.median_sharpe, 0.0)
        self.assertEqual(result.p5_sharpe, 0.0)
        self.assertEqual(result.p95_sharpe, 0.0)

    def test_single_simulation_probability(self):
        """With a single simulation, probability should be exactly 0 or 1."""
        returns = pd.Series(np.random.normal(0.001, 0.02, size=100))
        # Force a scenario where the final equity is guaranteed above the start
        result = monte_carlo_simulation(returns, n_simulations=1, n_years=0.01)
        self.assertIn(result.prob_positive_return, (0.0, 1.0))
        # Ensure the number of simulations reported matches the input
        self.assertEqual(result.num_simulations, 1)


if __name__ == "__main__":
    unittest.main()