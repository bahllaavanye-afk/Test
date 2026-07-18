"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 100_000
DEFAULT_RANDOM_SEED = 42


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
    risk_free_daily: float = 0.05 / DAYS_PER_YEAR,
) -> MonteCarloResult:
    """Bootstrap daily returns to simulate *n_years* of equity paths.

    Args:
        daily_returns: Series of historical daily returns.
        n_simulations: Number of Monte Carlo paths to generate.
        n_years: Horizon in years for each simulated path.
        risk_free_daily: Daily risk‑free rate.

    Returns:
        MonteCarloResult containing summary statistics of the simulations.
    """
    n_days = n_years * DAYS_PER_YEAR
    returns_array = daily_returns.dropna().values
    sharpes: list[float] = []
    max_dds: list[float] = []
    positive = 0

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    for _ in range(n_simulations):
        sampled = rng.choice(returns_array, size=n_days, replace=True)
        equity = np.cumprod(1 + sampled) * INITIAL_CAPITAL
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = dd.min()

        excess = sampled - risk_free_daily
        sharpe = (
            (excess.mean() / excess.std() * np.sqrt(DAYS_PER_YEAR))
            if excess.std() > 0
            else 0.0
        )
        sharpes.append(sharpe)
        max_dds.append(max_dd)
        if equity[-1] > INITIAL_CAPITAL:
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