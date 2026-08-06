"""Monte Carlo simulation utilities.

This module provides a bootstrap Monte‑Carlo simulation for equity curves,
producing confidence intervals for Sharpe ratios and maximum drawdowns.
It is used by the back‑testing suite to assess strategy robustness under
different market realizations.
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonteCarloError(RuntimeError):
    """Raised when an unexpected error occurs during Monte Carlo simulation."""


@dataclass
class MonteCarloResult:
    """Aggregated statistics from a Monte‑Carlo simulation.

    Attributes
    ----------
    median_sharpe: float
        Median Sharpe ratio across all simulated paths.
    p5_sharpe: float
        5th percentile Sharpe ratio (unlucky tail).
    p95_sharpe: float
        95th percentile Sharpe ratio (lucky tail).
    median_max_dd: float
        Median maximum drawdown (negative value) across simulations.
    p95_max_dd: float
        95th percentile of maximum drawdown (the milder drawdown).
    p5_max_dd: float
        5th percentile of maximum drawdown (the severe drawdown).
    prob_positive_return: float
        Proportion of simulations that end with a positive return.
    num_simulations: int
        Number of Monte‑Carlo paths generated (not shown in repr).
    """

    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_max_dd: float
    p95_max_dd: float
    p5_max_dd: float
    prob_positive_return: float
    num_simulations: int = field(repr=False)

    def __post_init__(self) -> None:
        """Validate that all numeric fields contain real numbers and are within expected ranges."""
        numeric_fields = {
            "median_sharpe": self.median_sharpe,
            "p5_sharpe": self.p5_sharpe,
            "p95_sharpe": self.p95_sharpe,
            "median_max_dd": self.median_max_dd,
            "p95_max_dd": self.p95_max_dd,
            "p5_max_dd": self.p5_max_dd,
            "prob_positive_return": self.prob_positive_return,
        }
        for name, value in numeric_fields.items():
            if not isinstance(value, numbers.Real):
                raise ValueError(f"{name} must be a real number, got {type(value)}.")
        if not (0.0 <= self.prob_positive_return <= 1.0):
            raise ValueError(
                "prob_positive_return must be between 0 and 1 inclusive."
            )
        if not isinstance(self.num_simulations, int) or self.num_simulations <= 0:
            raise ValueError("num_simulations must be a positive integer.")


def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    n_years: int | float = 3,
    risk_free_daily: float = 0.05 / 252,
) -> MonteCarloResult:
    """Bootstrap daily returns to simulate N years of equity paths.

    Parameters
    ----------
    daily_returns : pd.Series
        Series of daily returns. Must be non‑empty, numeric, and contain finite values.
    n_simulations : int, default 1000
        Number of Monte‑Carlo paths to generate. Must be a positive integer.
    n_years : int or float, default 3
        Number of years to simulate. Must be a positive number.
    risk_free_daily : float, default 0.05/252
        Daily risk‑free rate. Must be a real number.

    Returns
    -------
    MonteCarloResult
        Aggregated statistics from the simulations.

    Raises
    ------
    ValueError
        If any input is invalid.
    MonteCarloError
        If an unexpected error occurs during the simulation.
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

    n_days: int = int(n_years * 252)
    returns_array: np.ndarray = daily_returns.dropna().values
    sharpes: List[float] = []
    max_dds: List[float] = []
    positive: int = 0

    rng = np.random.default_rng(42)

    try:
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
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "Unexpected error during Monte Carlo simulation",
            extra={"n_simulations": n_simulations, "n_years": n_years},
        )
        raise MonteCarloError("Monte Carlo simulation failed") from exc

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