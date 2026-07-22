"""Monte Carlo simulation utilities for bootstrapping equity curves.

This module provides a simple Monte Carlo bootstrap that resamples historical daily
returns to generate synthetic equity paths. The resulting statistics are useful for
estimating the robustness of a strategy's Sharpe ratio and maximum drawdown under
different market realizations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation summary statistics.

    Attributes
    ----------
    median_sharpe: float
        Median Sharpe ratio across all simulated paths.
    p5_sharpe: float
        5th percentile Sharpe ratio.
    p95_sharpe: float
        95th percentile Sharpe ratio.
    median_max_dd: float
        Median maximum drawdown (as a negative number) across simulations.
    p95_max_dd: float
        95th percentile maximum drawdown.
    prob_positive_return: float
        Probability that the final portfolio value exceeds the initial capital.
    num_simulations: int
        Number of Monte Carlo paths that were generated.
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
    """Bootstrap daily returns to simulate multi‑year equity paths.

    Parameters
    ----------
    daily_returns : pd.Series
        Historical daily returns of the strategy (e.g., percentage returns). NaN values
        are ignored.
    n_simulations : int, default 1000
        Number of Monte Carlo paths to generate.
    n_years : int, default 3
        Horizon of each simulated path expressed in years. The function assumes 252
        trading days per year.
    risk_free_daily : float, default 0.05 / 252
        Daily risk‑free rate used to compute excess returns for Sharpe ratio calculation.

    Returns
    -------
    MonteCarloResult
        A dataclass instance containing summary statistics of the simulated paths,
        including median and percentile Sharpe ratios, drawdown metrics, and the
        probability of a positive return.
    """
    n_days = n_years * 252
    returns_array = daily_returns.dropna().values
    sharpes: list[float] = []
    max_dds: list[float] = []
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