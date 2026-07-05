"""
Drawdown recovery time estimator.
Given current drawdown and historical avg daily return, estimate when portfolio recovers.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from datetime import date, timedelta


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


def estimate_recovery(
    returns: list[float],
    current_drawdown: float,
) -> RecoveryEstimate:
    """
    Monte Carlo estimate of drawdown recovery time.

    Args:
        returns: Historical daily returns list
        current_drawdown: Current drawdown as fraction (e.g. 0.05 = 5% below peak)
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
        # Negative or zero drift — unlikely to recover
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
    first_recovery = []
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


# Unit tests for edge cases
if __name__ == "__main__":
    import unittest

    class TestEstimateRecoveryEdgeCases(unittest.TestCase):
        def test_empty_returns_or_nonpositive_drawdown(self):
            # Empty returns list
            result = estimate_recovery([], 0.05)
            self.assertEqual(result.current_drawdown_pct, 0)
            self.assertEqual(result.avg_daily_return, 0)
            self.assertEqual(result.expected_recovery_days, 0)
            self.assertIsNotNone(result.expected_recovery_date)
            self.assertEqual(result.probability_recover_30d, 1.0)
            self.assertEqual(result.probability_recover_90d, 1.0)

            # Non‑positive drawdown
            result = estimate_recovery([0.01, 0.02], -0.01)
            self.assertEqual(result.current_drawdown_pct, 0)
            self.assertEqual(result.expected_recovery_days, 0)

        def test_negative_average_return(self):
            # All negative returns lead to mu <= 0 branch
            returns = [-0.01, -0.02, -0.015]
            result = estimate_recovery(returns, 0.05)
            self.assertAlmostEqual(result.current_drawdown_pct, 0.05)
            self.assertLessEqual(result.avg_daily_return, 0)
            self.assertIsNone(result.expected_recovery_days)
            self.assertIsNone(result.expected_recovery_date)
            self.assertAlmostEqual(result.probability_recover_30d, 0.1)
            self.assertAlmostEqual(result.probability_recover_90d, 0.25)

        def test_zero_average_return(self):
            # Zero mean (mu == 0) should also trigger the non‑recovery branch
            returns = [0.01, -0.01, 0.0]
            result = estimate_recovery(returns, 0.02)
            self.assertAlmostEqual(result.current_drawdown_pct, 0.02)
            self.assertLessEqual(result.avg_daily_return, 0)  # mu is effectively 0
            self.assertIsNone(result.expected_recovery_days)
            self.assertIsNone(result.expected_recovery_date)

        def test_positive_average_return(self):
            # Positive drift should produce a finite recovery estimate
            returns = [0.001, 0.002, 0.003, 0.0015]
            result = estimate_recovery(returns, 0.01)
            self.assertGreater(result.expected_recovery_days, 0)
            self.assertIsInstance(result.expected_recovery_date, date)
            self.assertGreaterEqual(result.probability_recover_30d, 0.0)
            self.assertLessEqual(result.probability_recover_30d, 1.0)
            self.assertGreaterEqual(result.probability_recover_90d, 0.0)
            self.assertLessEqual(result.probability_recover_90d, 1.0)

    unittest.main(argv=["first-arg-is-ignored"], exit=False)