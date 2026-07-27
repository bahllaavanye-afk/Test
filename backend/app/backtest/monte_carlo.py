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

    rng = np.random.default_rng(42)

    # Vectorized sampling of returns for all simulations at once
    sampled = rng.choice(returns_array, size=(n_simulations, n_days), replace=True)

    # Equity curve computation
    equity = np.cumprod(1 + sampled, axis=1) * 100_000

    # Maximum drawdown per simulation
    peak = np.maximum.accumulate(equity, axis=1)
    dd = (equity - peak) / peak
    max_dds = dd.min(axis=1)

    # Sharpe ratio per simulation
    excess = sampled - risk_free_daily
    excess_std = excess.std(axis=1, ddof=0)
    # Avoid division by zero
    sharpe_vals = np.where(
        excess_std > 0,
        excess.mean(axis=1) / excess_std * np.sqrt(252),
        0.0,
    )

    # Probability of positive return
    positive = np.sum(equity[:, -1] > 100_000)

    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpe_vals)), 4),
        p5_sharpe=round(float(np.percentile(sharpe_vals, 5)), 4),
        p95_sharpe=round(float(np.percentile(sharpe_vals, 95)), 4),
        median_max_dd=round(float(np.median(max_dds)), 4),
        p95_max_dd=round(float(np.percentile(max_dds, 95)), 4),
        prob_positive_return=round(positive / n_simulations, 4),
        num_simulations=n_simulations,
    )