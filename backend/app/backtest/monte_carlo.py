"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import numbers
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Configuration constants – tweak to tighten entry/exit criteria
# ----------------------------------------------------------------------
ENTRY_WINDOW_DAYS: int = 20          # Minimum look‑back for entry confirmation
ENTRY_MEAN_THRESHOLD: float = 0.0   # Entry requires positive mean return over the window
EARLY_EXIT_DD: float = 0.20         # Stop simulation if equity falls 20% below initial capital
INITIAL_CAPITAL: float = 100_000.0  # Starting equity for each simulation
MAX_ENTRY_RESAMPLES: int = 10       # Guard against endless resampling loops
RANDOM_SEED: int = 42               # Deterministic seed for reproducibility


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
    """Perform lightweight validation of public arguments."""
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


def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    n_years: int = 3,
    risk_free_daily: float = 0.05 / 252,
) -> MonteCarloResult:
    """Bootstrap daily returns to simulate N years of paths.

    The simulation incorporates tighter entry confirmation (positive mean over a short
    window) and an early‑exit rule based on drawdown to produce more realistic performance
    distributions.

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
    """
    _validate_inputs(daily_returns, n_simulations, n_years, risk_free_daily)

    n_days = int(n_years * 252)
    returns_array = daily_returns.dropna().values

    sharpes: list[float] = []
    max_dds: list[float] = []
    positive_count = 0

    rng = np.random.default_rng(RANDOM_SEED)

    for _ in range(n_simulations):
        # ------------------------------------------------------------------
        # Entry filter: ensure the first ENTRY_WINDOW_DAYS have a positive mean.
        # ------------------------------------------------------------------
        attempts = 0
        sampled = rng.choice(returns_array, size=n_days, replace=True)
        while sampled[:ENTRY_WINDOW_DAYS].mean() <= ENTRY_MEAN_THRESHOLD and attempts < MAX_ENTRY_RESAMPLES:
            sampled = rng.choice(returns_array, size=n_days, replace=True)
            attempts += 1

        # ------------------------------------------------------------------
        # Build equity curve with early‑exit drawdown protection.
        # ------------------------------------------------------------------
        equity = np.empty_like(sampled, dtype=np.float64)
        equity[0] = INITIAL_CAPITAL * (1 + sampled[0])

        early_exit_index = None
        for i in range(1, len(sampled)):
            equity[i] = equity[i - 1] * (1 + sampled[i])
            if equity[i] < INITIAL_CAPITAL * (1 - EARLY_EXIT_DD):
                early_exit_index = i
                break

        if early_exit_index is not None:
            equity = equity[: early_exit_index + 1]

        # ------------------------------------------------------------------
        # Performance metrics for the simulated path.
        # ------------------------------------------------------------------
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        excess = sampled[: len(equity)] - risk_free_daily
        sharpe = (
            (excess.mean() / excess.std() * np.sqrt(252))
            if excess.std() > 0
            else 0.0
        )

        sharpes.append(sharpe)
        max_dds.append(max_dd)

        if equity[-1] > INITIAL_CAPITAL:
            positive_count += 1

    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpes)), 4),
        p5_sharpe=round(float(np.percentile(sharpes, 5)), 4),
        p95_sharpe=round(float(np.percentile(sharpes, 95)), 4),
        median_max_dd=round(float(np.median(max_dds)), 4),
        p95_max_dd=round(float(np.percentile(max_dds, 95)), 4),
        prob_positive_return=round(positive_count / n_simulations, 4),
        num_simulations=n_simulations,
    )