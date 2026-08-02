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
    # NOTE ON SIGN: max_dd is `dd.min()`, so drawdowns are NEGATIVE. The 95th
    # percentile of a negative series is therefore the MILDEST drawdown, not the
    # worst — `p95_max_dd` is the lucky tail, despite reading like a risk
    # number. It is kept for compatibility and left as-is.
    #
    # `p5_max_dd` is the severe tail, and is the one to quote as risk. This
    # matches `p5_sharpe`, which is already the unlucky end.
    p95_max_dd: float
    p5_max_dd: float
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
        p5_max_dd=round(float(np.percentile(max_dds, 5)), 4),
        prob_positive_return=round(positive / n_simulations, 4),
        num_simulations=n_simulations,
    )


# --------------------------------------------------------------------------- #
# Unit tests for edge‑case handling
# --------------------------------------------------------------------------- #

import pytest


def test_single_value_series_minimum_simulation():
    """Single‑value daily returns with the smallest allowed simulation count."""
    daily_returns = pd.Series([0.01])  # constant 1% daily return
    result = monte_carlo_simulation(
        daily_returns=daily_returns,
        n_simulations=1,
        n_years=1,  # 252 trading days
    )
    # With constant positive returns, Sharpe should be 0 (std = 0) and drawdown 0.
    assert result.median_sharpe == 0.0
    assert result.p5_sharpe == 0.0
    assert result.p95_sharpe == 0.0
    assert result.median_max_dd == 0.0
    assert result.p5_max_dd == 0.0
    assert result.p95_max_dd == 0.0
    # Final equity is > initial capital, so probability of positive return is 1.
    assert result.prob_positive_return == 1.0
    assert result.num_simulations == 1


def test_small_fraction_of_year_boundary():
    """Very small n_years leading to a tiny number of simulated days."""
    daily_returns = pd.Series([0.0, -0.01, 0.02])
    # 0.01 years ≈ 2.52 days → int conversion yields 2 days.
    result = monte_carlo_simulation(
        daily_returns=daily_returns,
        n_simulations=10,
        n_years=0.01,
    )
    # Ensure the function runs and returns sensible bounds.
    assert isinstance(result.median_sharpe, float)
    assert 0.0 <= result.prob_positive_return <= 1.0
    assert result.num_simulations == 10
    # Median drawdown should be between the theoretical extremes.
    assert result.median_max_dd <= 0.0  # drawdowns are non‑positive


def test_non_integer_n_years_and_large_simulations():
    """Non‑integer years and a larger simulation count to test input flexibility."""
    daily_returns = pd.Series(np.random.normal(0.0005, 0.01, size=1000))
    result = monte_carlo_simulation(
        daily_returns=daily_returns,
        n_simulations=500,
        n_years=2.5,  # non‑integer years
    )
    # Basic sanity checks
    assert result.num_simulations == 500
    assert 0.0 <= result.prob_positive_return <= 1.0
    # Percentiles must be ordered correctly.
    assert result.p5_sharpe <= result.median_sharpe <= result.p95_sharpe
    assert result.p5_max_dd <= result.median_max_dd <= result.p95_max_dd
    # Ensure no NaNs appear in the result fields.
    for field in result.__dict__.values():
        assert not np.isnan(field)