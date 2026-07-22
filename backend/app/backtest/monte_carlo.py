"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_max_dd: float
    p95_max_dd: float
    prob_positive_return: float
    num_simulations: int


def _validate_inputs(
    daily_returns: pd.Series,
    n_simulations: int,
    n_years: int,
    risk_free_daily: float,
) -> None:
    """Validate function arguments and raise informative exceptions."""
    if not isinstance(daily_returns, pd.Series):
        raise TypeError(
            f"daily_returns must be a pandas Series, got {type(daily_returns).__name__}"
        )
    if daily_returns.empty:
        raise ValueError("daily_returns Series is empty; cannot perform simulation.")
    if not isinstance(n_simulations, int):
        raise TypeError(
            f"n_simulations must be an int, got {type(n_simulations).__name__}"
        )
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    if not isinstance(n_years, int):
        raise TypeError(f"n_years must be an int, got {type(n_years).__name__}")
    if n_years <= 0:
        raise ValueError("n_years must be a positive integer.")
    if not isinstance(risk_free_daily, (int, float, np.floating, np.integer)):
        raise TypeError(
            f"risk_free_daily must be a numeric type, got {type(risk_free_daily).__name__}"
        )


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
        Historical daily returns to sample from.
    n_simulations : int, optional
        Number of Monte Carlo paths to generate, by default 1000.
    n_years : int, optional
        Horizon in years for each simulated path, by default 3.
    risk_free_daily : float, optional
        Daily risk‑free rate used for Sharpe ratio calculation, by default 0.05/252.

    Returns
    -------
    MonteCarloResult
        Aggregated statistics across all simulated paths.

    Raises
    ------
    TypeError, ValueError
        If input validation fails.
    RuntimeError
        If an unexpected error occurs during simulation.
    """
    try:
        _validate_inputs(daily_returns, n_simulations, n_years, risk_free_daily)
    except (TypeError, ValueError) as exc:
        logger.error("Input validation failed: %s", exc, exc_info=True)
        raise

    n_days = n_years * 252
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
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected error during Monte Carlo simulation: %s", exc)
        raise RuntimeError("Monte Carlo simulation failed") from exc

    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpes)), 4),
        p5_sharpe=round(float(np.percentile(sharpes, 5)), 4),
        p95_sharpe=round(float(np.percentile(sharpes, 95)), 4),
        median_max_dd=round(float(np.median(max_dds)), 4),
        p95_max_dd=round(float(np.percentile(max_dds, 95)), 4),
        prob_positive_return=round(positive / n_simulations, 4),
        num_simulations=n_simulations,
    )