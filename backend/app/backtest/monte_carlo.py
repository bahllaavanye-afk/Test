"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional


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
    if not isinstance(daily_returns, pd.Series):
        raise TypeError("daily_returns must be a pandas Series.")
    if daily_returns.empty:
        raise ValueError("daily_returns cannot be empty.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    if n_years <= 0:
        raise ValueError("n_years must be a positive integer.")
    if not isinstance(risk_free_daily, (float, int)):
        raise TypeError("risk_free_daily must be a numeric type.")


def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    n_years: int = 3,
    risk_free_daily: float = 0.05 / 252,
    *,
    entry_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    stop_loss: Optional[float] = None,
    seed: Optional[int] = 42,
) -> MonteCarloResult:
    """Bootstrap daily returns to simulate N years of paths.

    Args:
        daily_returns: Series of historical daily returns.
        n_simulations: Number of Monte‑Carlo paths to generate.
        n_years: Horizon in years for each simulated path.
        risk_free_daily: Daily risk‑free rate used for Sharpe calculation.
        entry_filter: Optional callable that receives the sampled returns array
            and returns a boolean mask. Only returns where the mask is True are
            used for the path, tightening entry conditions.
        stop_loss: Optional fractional loss (e.g., 0.2 for 20 %). If provided,
            the simulation stops the path when equity falls below
            ``initial_capital * (1 - stop_loss)``.
        seed: Random seed for reproducibility. If ``None`` a nondeterministic
            generator is used.

    Returns:
        MonteCarloResult containing aggregated performance metrics.
    """
    _validate_inputs(daily_returns, n_simulations, n_years, risk_free_daily)

    n_days = n_years * 252
    returns_array = daily_returns.dropna().values
    initial_capital = 100_000.0

    rng = np.random.default_rng(seed)

    sharpes: list[float] = []
    max_dds: list[float] = []
    positive = 0

    for _ in range(n_simulations):
        sampled = rng.choice(returns_array, size=n_days, replace=True)

        # Apply optional entry filter
        if entry_filter is not None:
            mask = entry_filter(sampled)
            if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                raise ValueError("entry_filter must return a boolean numpy array.")
            # Ensure at least one true value to avoid empty equity series
            if not mask.any():
                # Fallback to original sample if filter removes everything
                filtered = sampled
            else:
                filtered = sampled[mask]
            # If filtered length differs, repeat to reach n_days
            if filtered.size < n_days:
                repeats = int(np.ceil(n_days / filtered.size))
                filtered = np.tile(filtered, repeats)[:n_days]
            sampled = filtered

        equity = np.empty_like(sampled)
        equity[0] = initial_capital * (1 + sampled[0])
        stop_triggered = False

        for i in range(1, sampled.size):
            equity[i] = equity[i - 1] * (1 + sampled[i])
            if stop_loss is not None and not stop_triggered:
                if equity[i] <= initial_capital * (1 - stop_loss):
                    # Stop loss hit: freeze equity at the stop level for remaining steps
                    equity[i:] = equity[i]
                    stop_triggered = True

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
        if equity[-1] > initial_capital:
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