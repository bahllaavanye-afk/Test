"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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
        Series of daily returns (as decimal, e.g., 0.001 for 0.1%).
    n_simulations : int, optional
        Number of Monte‑Carlo paths to generate. Must be positive.
    n_years : int, optional
        Horizon in years for each simulated path. Must be positive.
    risk_free_daily : float, optional
        Daily risk‑free rate used to compute excess returns.

    Returns
    -------
    MonteCarloResult
        Summary statistics of the simulated Sharpe ratios and maximum drawdowns.

    Raises
    ------
    ValueError
        If ``daily_returns`` is empty after dropping NaNs, or if ``n_simulations``
        or ``n_years`` are not positive.
    """
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer")
    if n_years <= 0:
        raise ValueError("n_years must be a positive integer")

    # Clean input series
    returns_array = daily_returns.dropna().values
    if returns_array.size == 0:
        raise ValueError("daily_returns contains no valid data after dropping NaNs")

    n_days = n_years * 252
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


# ==============================
# Unit tests for edge conditions
# ==============================

def test_empty_series_raises():
    empty_series = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="daily_returns contains no valid data"):
        monte_carlo_simulation(empty_series, n_simulations=10, n_years=1)


def test_constant_returns_zero_sharpe_and_drawdown():
    # Constant positive return; zero variance means Sharpe should be 0,
    # and there should be no drawdown.
    const_ret = pd.Series([0.001] * 252)  # one year of constant 0.1% daily return
    result = monte_carlo_simulation(const_ret, n_simulations=20, n_years=1)
    assert result.median_sharpe == 0.0
    assert result.p5_sharpe == 0.0
    assert result.p95_sharpe == 0.0
    # No drawdown because equity only moves upward
    assert result.median_max_dd == 0.0
    assert result.p95_max_dd == 0.0
    # With a positive deterministic drift, all simulations end above start capital
    assert result.prob_positive_return == 1.0


def test_invalid_simulation_count_raises():
    series = pd.Series([0.01, -0.01, 0.005])
    with pytest.raises(ValueError, match="n_simulations must be a positive integer"):
        monte_carlo_simulation(series, n_simulations=0, n_years=1)


def test_invalid_years_raises():
    series = pd.Series([0.01, -0.01, 0.005])
    with pytest.raises(ValueError, match="n_years must be a positive integer"):
        monte_carlo_simulation(series, n_simulations=10, n_years=0)