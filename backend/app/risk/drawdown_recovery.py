"""
Drawdown recovery time estimator.

Provides utilities to estimate the number of days required for a portfolio to
recover from its current drawdown based on historical daily returns. The core
function uses a simple analytical estimate combined with a Monte‑Carlo simulation
to produce a median recovery time, a recovery date, and probabilities of
recovery within 30 and 90 days.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict


@dataclass
class RecoveryEstimate:
    """
    Container for the results of a drawdown recovery estimation.

    Attributes
    ----------
    current_drawdown_pct: float
        The current drawdown expressed as a fraction of the portfolio value
        (e.g., ``0.05`` for a 5 % drawdown).
    avg_daily_return: float
        The average daily return of the historical series used for the
        estimation, expressed as a fraction.
    expected_recovery_days: int | None
        The median number of days required to recover, or ``None`` if recovery
        is deemed unlikely.
    expected_recovery_date: date | None
        The calendar date on which recovery is expected, or ``None`` if not
        applicable.
    probability_recover_30d: float
        Estimated probability of recovery within 30 days (0 – 1).
    probability_recover_90d: float
        Estimated probability of recovery within 90 days (0 – 1).
    """

    current_drawdown_pct: float
    avg_daily_return: float
    expected_recovery_days: int | None
    expected_recovery_date: date | None
    probability_recover_30d: float
    probability_recover_90d: float

    def to_dict(self) -> Dict[str, object]:
        """
        Convert the estimate to a serialisable dictionary.

        Returns
        -------
        dict
            A dictionary with human‑readable, rounded values suitable for JSON
            output or logging.
        """
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


def estimate_recovery(
    returns: List[float],
    current_drawdown: float,
) -> RecoveryEstimate:
    """
    Estimate the time required for a portfolio to recover from its current drawdown.

    The function first checks for trivial or impossible cases (empty return series
    or non‑positive drawdown). If the average historical return is non‑positive,
    a pessimistic estimate is returned. Otherwise, a simple analytical estimate
    is calculated, and a Monte‑Carlo simulation of 1 000 paths (up to 365 days)
    provides a median recovery horizon and probabilities of recovery within
    30 and 90 days.

    Parameters
    ----------
    returns : List[float]
        Historical daily returns expressed as fractions (e.g., ``0.001`` for
        0.1 % daily return). The list must contain at least one element for a
        meaningful estimate.
    current_drawdown : float
        Current drawdown expressed as a fraction of the portfolio value
        (positive values indicate a loss below the peak).

    Returns
    -------
    RecoveryEstimate
        A populated :class:`RecoveryEstimate` instance containing the estimated
        recovery horizon, recovery date, and short‑term recovery probabilities.
    """
    if not returns or current_drawdown <= 0:
        return RecoveryEstimate(
            current_drawdown_pct=0,
            avg_daily_return=0,
            expected_recovery_days=0,
            expected_recovery_date=date.today(),
            probability_recover_30d=1.0,
            probability_recover_90d=1.0,
        )

    arr = np.array(returns)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))

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

    # Simple estimate: days = drawdown / avg_daily_return
    naive_days = int(current_drawdown / (mu + 1e-9))

    # Monte Carlo: simulate 1000 paths, check how many recover within N days
    n_sims = 1000
    max_days = 365
    np.random.seed(None)
    sim_returns = np.random.normal(mu, sigma, (n_sims, max_days))
    cum = np.cumprod(1 + sim_returns, axis=1) - 1  # cumulative return from today

    target = current_drawdown  # need to gain this much to recover
    recover_30 = float(np.mean(np.any(cum[:, :30] >= target, axis=1)))
    recover_90 = float(np.mean(np.any(cum[:, :90] >= target, axis=1)))

    # Median recovery time across simulations
    first_recovery: List[int] = []
    for path in cum:
        idx = np.argmax(path >= target)
        if path[idx] >= target:
            first_recovery.append(idx + 1)
    median_days = int(np.median(first_recovery)) if first_recovery else naive_days

    recovery_date = date.today() + timedelta(days=median_days)
    return RecoveryEstimate(
        current_drawdown_pct=current_drawdown,
        avg_daily_return=mu,
        expected_recovery_days=median_days,
        expected_recovery_date=recovery_date,
        probability_recover_30d=recover_30,
        probability_recover_90d=recover_90,
    )