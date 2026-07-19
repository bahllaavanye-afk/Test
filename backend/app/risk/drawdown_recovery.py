"""
Drawdown recovery time estimator.
Given current drawdown and historical average daily return, estimate when portfolio recovers.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List


@dataclass
class RecoveryEstimate:
    current_drawdown_pct: float
    avg_daily_return: float
    expected_recovery_days: int | None
    expected_recovery_date: date | None
    probability_recover_30d: float
    probability_recover_90d: float

    def to_dict(self) -> dict:
        """Serialize the estimate to a JSON‑friendly dictionary."""
        return {
            "current_drawdown_pct": round(self.current_drawdown_pct * 100, 2),
            "avg_daily_return_pct": round(self.avg_daily_return * 100, 3),
            "expected_recovery_days": self.expected_recovery_days,
            "expected_recovery_date": (
                self.expected_recovery_date.isoformat()
                if self.expected_recovery_date
                else None
            ),
            "probability_recover_30d": round(self.probability_recover_30d, 3),
            "probability_recover_90d": round(self.probability_recover_90d, 3),
        }


def _validate_inputs(returns: List[float], current_drawdown: float) -> bool:
    """Return True if inputs are sufficient for a meaningful estimate."""
    # Require a non‑empty return series and a positive drawdown.
    if not returns or current_drawdown <= 0:
        return False
    # Require a minimum history to compute reliable statistics.
    if len(returns) < 30:
        return False
    return True


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
        RecoveryEstimate containing expected recovery horizon and probabilities.
    """
    # Quick exit for insufficient data or non‑positive drawdown.
    if not _validate_inputs(returns, current_drawdown):
        return RecoveryEstimate(
            current_drawdown_pct=0.0,
            avg_daily_return=0.0,
            expected_recovery_days=0,
            expected_recovery_date=date.today(),
            probability_recover_30d=1.0,
            probability_recover_90d=1.0,
        )

    arr = np.asarray(returns, dtype=float)

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))

    # If the drift is non‑positive, recovery is highly unlikely.
    if mu <= 0:
        return RecoveryEstimate(
            current_drawdown_pct=current_drawdown,
            avg_daily_return=mu,
            expected_recovery_days=None,
            expected_recovery_date=None,
            probability_recover_30d=0.1,
            probability_recover_90d=0.25,
        )

    # Simple deterministic estimate as a fallback.
    naive_days = max(int(current_drawdown / (mu + 1e-9)), 1)

    # Monte‑Carlo simulation parameters.
    n_sims = 2000
    max_days = 365

    rng = np.random.default_rng()
    sim_returns = rng.normal(mu, sigma, size=(n_sims, max_days))

    # Compute cumulative return series for each simulation.
    cum = np.cumprod(1 + sim_returns, axis=1) - 1  # cumulative return from today

    target = current_drawdown  # Need to gain this amount to recover.

    # Probabilities of recovery within 30 and 90 days.
    recover_30 = float(np.mean(np.any(cum[:, :30] >= target, axis=1)))
    recover_90 = float(np.mean(np.any(cum[:, :90] >= target, axis=1)))

    # Find first day of recovery for each path.
    first_recovery_days = []
    for path in cum:
        # np.where returns indices where condition holds; take the first.
        hits = np.where(path >= target)[0]
        if hits.size > 0:
            first_recovery_days.append(int(hits[0]) + 1)  # +1 for 1‑based day count

    # Use median of successful recoveries; fall back to naive estimate if none.
    if first_recovery_days:
        median_days = int(np.median(first_recovery_days))
    else:
        median_days = naive_days

    # Guard against extreme outliers (e.g., median > max_days).
    median_days = min(median_days, max_days)

    recovery_date = date.today() + timedelta(days=median_days)

    return RecoveryEstimate(
        current_drawdown_pct=current_drawdown,
        avg_daily_return=mu,
        expected_recovery_days=median_days,
        expected_recovery_date=recovery_date,
        probability_recover_30d=recover_30,
        probability_recover_90d=recover_90,
    )