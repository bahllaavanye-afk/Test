"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations
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
    """Bootstrap daily returns to simulate N years of paths."""
    n_days = n_years * 252
    returns_array = daily_returns.dropna().values
    if returns_array.size == 0:
        raise ValueError("daily_returns contains no valid data after dropping NaNs.")
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


# Unit tests for edge cases
import unittest


class TestMonteCarloSimulation(unittest.TestCase):
    def test_constant_returns_sharpe_zero(self):
        """Constant returns should yield Sharpe ratio of 0 due to zero volatility."""
        const_return = pd.Series([0.001] * 252)  # one year of constant daily return
        result = monte_carlo_simulation(const_return, n_simulations=10, n_years=1)
        self.assertEqual(result.median_sharpe, 0.0)
        self.assertTrue(0.0 <= result.prob_positive_return <= 1.0)

    def test_empty_series_raises(self):
        """An empty series (or all NaNs) should raise a ValueError."""
        empty_series = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            monte_carlo_simulation(empty_series, n_simulations=10, n_years=1)

        nan_series = pd.Series([np.nan, np.nan])
        with self.assertRaises(ValueError):
            monte_carlo_simulation(nan_series, n_simulations=10, n_years=1)

    def test_single_simulation(self):
        """Running a single simulation should still produce valid statistics."""
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        result = monte_carlo_simulation(returns, n_simulations=1, n_years=1)
        # With one simulation, median and percentiles are the same as the single sample
        self.assertAlmostEqual(result.median_sharpe, result.p5_sharpe, places=4)
        self.assertAlmostEqual(result.median_sharpe, result.p95_sharpe, places=4)
        self.assertAlmostEqual(result.median_max_dd, result.p95_max_dd, places=4)
        self.assertIn(result.prob_positive_return, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()