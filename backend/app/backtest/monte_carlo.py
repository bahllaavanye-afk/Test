"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import numbers
from typing import List, Tuple

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


def _validate_inputs(
    daily_returns: pd.Series,
    n_simulations: int,
    n_years: int | float,
    risk_free_daily: float,
) -> None:
    """Validate inputs for the Monte Carlo simulation."""
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


def _simulate_path(
    rng: np.random.Generator,
    returns_array: np.ndarray,
    n_days: int,
    risk_free_daily: float,
    initial_capital: float = 100_000.0,
) -> Tuple[float, float, bool]:
    """Run a single Monte Carlo path and return Sharpe, max drawdown, and positivity."""
    sampled = rng.choice(returns_array, size=n_days, replace=True)
    equity = np.cumprod(1 + sampled) * initial_capital

    # Maximum drawdown (negative value)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Sharpe ratio
    excess = sampled - risk_free_daily
    if excess.std() > 0:
        sharpe = excess.mean() / excess.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    positive = equity[-1] > initial_capital
    return sharpe, max_dd, positive


def _aggregate_results(
    sharpes: List[float],
    max_dds: List[float],
    positive_count: int,
    n_simulations: int,
) -> MonteCarloResult:
    """Create a MonteCarloResult from collected simulation statistics."""
    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpes)), 4),
        p5_sharpe=round(float(np.percentile(sharpes, 5)), 4),
        p95_sharpe=round(float(np.percentile(sharpes, 95)), 4),
        median_max_dd=round(float(np.median(max_dds)), 4),
        p95_max_dd=round(float(np.percentile(max_dds, 95)), 4),
        p5_max_dd=round(float(np.percentile(max_dds, 5)), 4),
        prob_positive_return=round(positive_count / n_simulations, 4),
        num_simulations=n_simulations,
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
    _validate_inputs(daily_returns, n_simulations, n_years, risk_free_daily)

    n_days = int(n_years * 252)
    returns_array = daily_returns.dropna().values

    sharpes: List[float] = []
    max_dds: List[float] = []
    positive_count = 0

    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        sharpe, max_dd, positive = _simulate_path(
            rng, returns_array, n_days, risk_free_daily
        )
        sharpes.append(sharpe)
        max_dds.append(max_dd)
        if positive:
            positive_count += 1

    return _aggregate_results(sharpes, max_dds, positive_count, n_simulations)