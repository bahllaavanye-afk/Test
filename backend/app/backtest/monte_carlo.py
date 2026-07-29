"""Monte Carlo simulation utilities for bootstrapping equity curves.

This module provides a single public function, :func:`monte_carlo_simulation`,
which generates Monte‑Carlo equity‑curve paths by resampling historical daily
returns. The function returns a :class:`MonteCarloResult` dataclass containing
summary statistics useful for assessing strategy robustness.
"""

from __future__ import annotations

import numbers
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    """Container for aggregated Monte‑Carlo simulation statistics.

    Attributes
    ----------
    median_sharpe: float
        Median annualised Sharpe ratio across all simulated paths.
    p5_sharpe: float
        5th percentile of the Sharpe ratio distribution.
    p95_sharpe: float
        95th percentile of the Sharpe ratio distribution.
    median_max_dd: float
        Median maximum drawdown (as a negative fraction) across simulations.
    p95_max_dd: float
        95th percentile of the maximum drawdown distribution.
    prob_positive_return: float
        Probability that the final portfolio value exceeds the initial capital.
    num_simulations: int
        Number of Monte‑Carlo paths that were generated.
    """

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
    """Bootstrap daily returns to simulate *n_years* of equity‑curve paths.

    The function resamples the supplied ``daily_returns`` with replacement,
    constructs a cumulative equity curve for each simulation, and computes a
    set of performance metrics. Results are aggregated into a
    :class:`MonteCarloResult` instance.

    Parameters
    ----------
    daily_returns : pd.Series
        Historical daily returns. Must be non‑empty, numeric, and contain only
        finite values. NaNs are dropped before resampling.
    n_simulations : int, default 1000
        Number of Monte‑Carlo paths to generate. Must be a positive integer.
    n_years : int, default 3
        Length of each simulated path expressed in years. Must be a positive
        number; the function assumes 252 trading days per year.
    risk_free_daily : float, default 0.05 / 252
        Daily risk‑free rate used to compute excess returns for the Sharpe
        ratio. Must be a real number.

    Returns
    -------
    MonteCarloResult
        Aggregated statistics from the simulations, including median and
        percentile Sharpe ratios, drawdowns, and the probability of a positive
        final return.

    Raises
    ------
    ValueError
        If any input fails validation (e.g., wrong type, empty series, or
        non‑finite values).
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