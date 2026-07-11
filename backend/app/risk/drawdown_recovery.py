"""
Drawdown recovery time estimator.
Given current drawdown and historical average daily return, estimate when portfolio recovers.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional


@dataclass
class RecoveryEstimate:
    current_drawdown_pct: float
    avg_daily_return: float
    expected_recovery_days: Optional[int]
    expected_recovery_date: Optional[date]
    probability_recover_30d: float
    probability_recover_90d: float

    def to_dict(self) -> dict:
        """Serialize the estimate for downstream consumption."""
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


def _clip_outliers(arr: np.ndarray, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> np.ndarray:
    """Clip extreme outliers to reduce distortion of mean/volatility estimates."""
    low, high = np.quantile(arr, [lower_quantile, upper_quantile])
    return np.clip(arr, low, high)


def estimate_recovery(
    returns: List[float],
    current_drawdown: float,
    *,
    n_sims: int = 2000,
    max_days: int = 365,
    seed: Optional[int] = None,
) -> RecoveryEstimate:
    """
    Monte‑Carlo estimate of drawdown recovery time.

    Parameters
    ----------
    returns: List[float]
        Historical daily simple returns (e.g. 0.001 for 0.1%).
    current_drawdown: float
        Current drawdown expressed as a positive fraction (e.g. 0.05 = 5% below peak).
    n_sims: int, optional
        Number of simulated paths. Default is 2000.
    max_days: int, optional
        Horizon for simulations. Default is 365 days.
    seed: int | None, optional
        Random seed for reproducibility. If None, system entropy is used.

    Returns
    -------
    RecoveryEstimate
        Contains naive and Monte‑Carlo based recovery metrics.
    """
    # Guard clauses – no data or non‑positive drawdown means immediate recovery.
    if not returns or current_drawdown <= 0:
        return RecoveryEstimate(
            current_drawdown_pct=0.0,
            avg_daily_return=0.0,
            expected_recovery_days=0,
            expected_recovery_date=date.today(),
            probability_recover_30d=1.0,
            probability_recover_90d=1.0,
        )

    # Convert to numpy array for vectorised ops.
    arr = np.asarray(returns, dtype=float)

    # Remove extreme outliers that could skew mu / sigma.
    arr = _clip_outliers(arr)

    # Use log‑returns for a more Gaussian‑like distribution.
    log_returns = np.log1p(arr)

    mu_log = float(np.mean(log_returns))
    sigma_log = float(np.std(log_returns, ddof=1))

    # Convert back to simple‑return drift for interpretation.
    mu = math.expm1(mu_log)  # ≈ expected simple return
    sigma = sigma_log  # volatility on log‑scale is appropriate for GBM simulation

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

    # Naïve deterministic estimate – serves as a fallback.
    naive_days = int(math.ceil(current_drawdown / (mu + 1e-12)))

    # Monte‑Carlo simulation using geometric Brownian motion.
    rng = np.random.default_rng(seed)
    # Daily log‑return draws.
    sim_log_returns = rng.normal(mu_log, sigma, size=(n_sims, max_days))
    # Cumulative log‑returns -> cumulative simple returns.
    cum_log = np.cumsum(sim_log_returns, axis=1)
    cum_simple = np.expm1(cum_log)  # shape (n_sims, max_days)

    # Recovery target expressed as simple return.
    target = current_drawdown

    # Probability of recovery within fixed horizons.
    recover_30 = float(np.mean(np.any(cum_simple[:, :30] >= target, axis=1)))
    recover_90 = float(np.mean(np.any(cum_simple[:, :90] >= target, axis=1)))

    # Determine first day of recovery for each path.
    first_recovery_days = []
    for path in cum_simple:
        # argmax returns first occurrence; if never true, argmax yields 0.
        recovered = path >= target
        if recovered.any():
            first_day = int(np.argmax(recovered) + 1)  # 1‑based day count
            first_recovery_days.append(first_day)

    # Median recovery time; fallback to naïve estimate if no path recovers.
    if first_recovery_days:
        median_days = int(round(np.median(first_recovery_days)))
        expected_date = date.today() + timedelta(days=median_days)
    else:
        median_days = naive_days
        expected_date = date.today() + timedelta(days=median_days)

    return RecoveryEstimate(
        current_drawdown_pct=current_drawdown,
        avg_daily_return=mu,
        expected_recovery_days=median_days,
        expected_recovery_date=expected_date,
        probability_recover_30d=min(max(recover_30, 0.0), 1.0),
        probability_recover_90d=min(max(recover_90, 0.0), 1.0),
    )