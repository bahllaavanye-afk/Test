"""Monte Carlo simulation utilities for bootstrapping equity curves.

This module provides a single public function, :func:`monte_carlo_simulation`, which
generates a distribution of possible equity‑curve outcomes by resampling historical
daily returns. The resulting statistics are encapsulated in a :class:`MonteCarloResult`
dataclass, offering median and percentile Sharpe ratios, drawdown metrics, and the
probability of achieving a positive return over the simulated horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    """Container for Monte‑Carlo simulation summary statistics.

    Attributes
    ----------
    median_sharpe: float
        Median Sharpe ratio across all simulated paths.
    p5_sharpe: float
        5th percentile Sharpe ratio.
    p95_sharpe: float
        95th percentile Sharpe ratio.
    median_max_dd: float
        Median maximum drawdown (as a negative fraction of peak equity).
    p95_max_dd: float
        95th percentile maximum drawdown.
    prob_positive_return: float
        Proportion of simulations that end with equity above the initial capital.
    num_simulations: int
        Number of Monte‑Carlo paths that were generated.
    """

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
    """Bootstrap daily returns to simulate *n_years* of equity‑curve paths.

    Parameters
    ----------
    daily_returns : pd.Series
        Historical daily total returns (e.g., P&L / capital). NaN values are ignored.
    n_simulations : int, default 1000
        Number of Monte‑Carlo trajectories to generate.
    n_years : int, default 3
        Length of each simulated path expressed in years. Assumes 252 trading days per year.
    risk_free_daily : float, default 0.05 / 252
        Daily risk‑free rate used to compute excess returns for Sharpe ratio calculation.

    Returns
    -------
    MonteCarloResult
        Summary statistics derived from the simulated equity curves.
    """
    n_days: int = n_years * 252
    returns_array: np.ndarray = daily_returns.dropna().values
    sharpes: list[float] = []
    max_dds: list[float] = []
    positive: int = 0

    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        sampled: np.ndarray = rng.choice(returns_array, size=n_days, replace=True)
        equity: np.ndarray = np.cumprod(1 + sampled) * 100_000
        peak: np.ndarray = np.maximum.accumulate(equity)
        dd: np.ndarray = (equity - peak) / peak
        max_dd: float = dd.min()

        excess: np.ndarray = sampled - risk_free_daily
        sharpe: float = (
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