"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

import pytest


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

    Args:
        daily_returns: Series of historical daily returns.
        n_simulations: Number of Monte‑Carlo paths to generate.
        n_years: Length of each simulated path in years.
        risk_free_daily: Daily risk‑free rate.

    Returns:
        MonteCarloResult containing summary statistics.

    Raises:
        ValueError: If ``daily_returns`` is empty or ``n_simulations`` is non‑positive.
    """
    if daily_returns.empty:
        raise ValueError("daily_returns must contain at least one value")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer")

    n_days = n_years * 252
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
        sharpe = (
            (excess.mean() / excess.std() * np.sqrt(252))
            if excess.std() > 0
            else 0.0
        )
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


# ===========================
# Unit tests for edge cases
# ===========================

def test_monte_carlo_empty_series_raises():
    """An empty return series should raise a ValueError."""
    empty_series = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="daily_returns must contain at least one value"):
        monte_carlo_simulation(empty_series, n_simulations=10)


def test_monte_carlo_nonpositive_simulations_raises():
    """A non‑positive number of simulations should raise a ValueError."""
    returns = pd.Series([0.001, -0.002, 0.003])
    with pytest.raises(ValueError, match="n_simulations must be a positive integer"):
        monte_carlo_simulation(returns, n_simulations=0)


def test_monte_carlo_constant_returns():
    """When all returns are identical, Sharpe should be zero and drawdown should be zero."""
    const_ret = pd.Series([0.001] * 252)  # one year of constant positive returns
    result = monte_carlo_simulation(const_ret, n_simulations=10, n_years=1)
    # Zero variance leads to Sharpe = 0
    assert result.median_sharpe == 0.0
    # No drawdown when equity only rises
    assert result.median_max_dd == 0.0
    # All simulations end above the initial capital
    assert result.prob_positive_return == 1.0


def test_monte_carlo_all_negative_returns():
    """All negative returns should yield a probability of positive return of zero."""
    neg_ret = pd.Series([-0.01] * 252)  # one year of -1% daily returns
    result = monte_carlo_simulation(neg_ret, n_simulations=10, n_years=1)
    # Since equity always declines, probability of ending above start is 0
    assert result.prob_positive_return == 0.0
    # Sharpe may be negative but should be computed without error
    assert isinstance(result.median_sharpe, float)