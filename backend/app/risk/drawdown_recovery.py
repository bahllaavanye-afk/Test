"""
Drawdown recovery time estimator.
Given current drawdown and historical avg daily return, estimate when portfolio recovers.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Tuple, Optional


@dataclass
class RecoveryEstimate:
    current_drawdown_pct: float
    avg_daily_return: float
    expected_recovery_days: int | None
    expected_recovery_date: date | None
    probability_recover_30d: float
    probability_recover_90d: float

    def to_dict(self) -> dict:
        return {
            "current_drawdown_pct": round(self.current_drawdown_pct * 100, 2),
            "avg_daily_return_pct": round(self.avg_daily_return * 100, 3),
            "expected_recovery_days": self.expected_recovery_days,
            "expected_recovery_date": self.expected_recovery_date.isoformat()
            if self.expected_recovery_date
            else None,
            "probability_recover_30d": round(self.probability_recover_30d, 3),
            "probability_recover_90d": round(self.probability_recover_90d, 3),
        }


def _validate_inputs(returns: List[float], current_drawdown: float) -> bool:
    """Return True if inputs are valid for estimation, False otherwise."""
    return bool(returns) and current_drawdown > 0


def _compute_statistics(arr: np.ndarray) -> Tuple[float, float]:
    """Calculate mean (mu) and unbiased standard deviation (sigma) of returns."""
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    return mu, sigma


def _simulate_paths(
    mu: float, sigma: float, n_sims: int = 1000, max_days: int = 365
) -> np.ndarray:
    """
    Simulate `n_sims` random return paths over `max_days` using a normal distribution.

    Returns a 2‑D array of cumulative returns (starting from 0) for each simulation.
    """
    np.random.seed(None)
    sim_returns = np.random.normal(mu, sigma, (n_sims, max_days))
    cum = np.cumprod(1 + sim_returns, axis=1) - 1
    return cum


def _calculate_recovery_probabilities(
    cum: np.ndarray, target: float
) -> Tuple[float, float]:
    """
    Compute the probability of recovering the target drawdown within 30 and 90 days.
    """
    recover_30 = float(np.mean(np.any(cum[:, :30] >= target, axis=1)))
    recover_90 = float(np.mean(np.any(cum[:, :90] >= target, axis=1)))
    return recover_30, recover_90


def _median_recovery_days(
    cum: np.ndarray, target: float, naive_days: int
) -> int:
    """
    Determine the median number of days required to recover the drawdown across simulations.
    Falls back to `naive_days` if no simulation recovers.
    """
    first_recovery: List[int] = []
    for path in cum:
        # idx is the first day where cumulative return meets/exceeds the target.
        idx = np.argmax(path >= target)
        if path[idx] >= target:
            first_recovery.append(idx + 1)  # days are 1‑based
    if first_recovery:
        return int(np.median(first_recovery))
    return naive_days


def estimate_recovery(
    returns: List[float],
    current_drawdown: float,
) -> RecoveryEstimate:
    """
    Monte Carlo estimate of drawdown recovery time.

    Args:
        returns: Historical daily returns list.
        current_drawdown: Current drawdown as a fraction (e.g. 0.05 = 5% below peak).

    Returns:
        RecoveryEstimate containing naive and Monte‑Carlo based recovery metrics.
    """
    if not _validate_inputs(returns, current_drawdown):
        return RecoveryEstimate(
            current_drawdown_pct=0,
            avg_daily_return=0,
            expected_recovery_days=0,
            expected_recovery_date=date.today(),
            probability_recover_30d=1.0,
            probability_recover_90d=1.0,
        )

    arr = np.array(returns)
    mu, sigma = _compute_statistics(arr)

    if mu <= 0:
        # Negative drift — unlikely to recover
        return RecoveryEstimate(
            current_drawdown_pct=current_drawdown,
            avg_daily_return=mu,
            expected_recovery_days=None,
            expected_recovery_date=None,
            probability_recover_30d=0.1,
            probability_recover_90d=0.25,
        )

    # Simple deterministic estimate
    naive_days = int(current_drawdown / (mu + 1e-9))

    # Monte Carlo simulation
    cum = _simulate_paths(mu, sigma, n_sims=1000, max_days=365)

    # Recovery probabilities
    prob_30d, prob_90d = _calculate_recovery_probabilities(cum, current_drawdown)

    # Median recovery time
    median_days = _median_recovery_days(cum, current_drawdown, naive_days)

    recovery_date = date.today() + timedelta(days=median_days)

    return RecoveryEstimate(
        current_drawdown_pct=current_drawdown,
        avg_daily_return=mu,
        expected_recovery_days=median_days,
        expected_recovery_date=recovery_date,
        probability_recover_30d=prob_30d,
        probability_recover_90d=prob_90d,
    )