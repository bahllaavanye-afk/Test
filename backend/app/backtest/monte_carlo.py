"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class MonteCarloError(RuntimeError):
    """Raised when an unexpected error occurs during Monte Carlo simulation."""


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

    n_days = int(n_years * 252)
    returns_array = daily_returns.dropna().values
    sharpes: list[float] = []
    max_dds: list[float] = []
    positive = 0

    rng = np.random.default_rng(42)

    try:
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