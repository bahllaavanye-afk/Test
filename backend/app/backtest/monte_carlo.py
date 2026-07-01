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
        Series of daily returns. Must contain at least one non‑NaN numeric value.
    n_simulations : int, default 1000
        Number of Monte Carlo paths to simulate. Must be a positive integer.
    n_years : int, default 3
        Horizon in years for each simulated path. Must be a positive integer.
    risk_free_daily : float, default 0.05/252
        Daily risk‑free rate used for Sharpe calculation. Must be a finite number.

    Returns
    -------
    MonteCarloResult
        Summary statistics of the simulated paths.

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    # Input validation
    if not isinstance(daily_returns, pd.Series):
        raise ValueError("daily_returns must be a pandas Series.")
    if daily_returns.dropna().empty:
        raise ValueError("daily_returns must contain at least one non-NaN value.")
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    if not isinstance(n_years, int) or n_years <= 0:
        raise ValueError("n_years must be a positive integer.")
    if not isinstance(risk_free_daily, numbers.Real):
        raise ValueError("risk_free_daily must be a numeric (real) value.")
    if not np.isfinite(risk_free_daily):
        raise ValueError("risk_free_daily must be a finite number.")

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