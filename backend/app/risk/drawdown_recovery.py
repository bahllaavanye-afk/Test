"""
Drawdown recovery time estimator.
Given current drawdown and historical avg daily return, estimate when portfolio recovers.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from datetime import date, timedelta

# Added imports for testing
import unittest
from unittest.mock import patch


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


# ---------------------------------------------------------------------------
# Unit tests for edge cases
# ---------------------------------------------------------------------------
class TestEstimateRecoveryEdgeCases(unittest.TestCase):
    def test_empty_returns(self):
        """Empty returns list should return zero days and full recovery probabilities."""
        result = estimate_recovery([], 0.05)
        self.assertEqual(result.expected_recovery_days, 0)
        self.assertEqual(result.expected_recovery_date, date.today())
        self.assertEqual(result.probability_recover_30d, 1.0)
        self.assertEqual(result.probability_recover_90d, 1.0)
        self.assertEqual(result.current_drawdown_pct, 0)
        self.assertEqual(result.avg_daily_return, 0)

    def test_negative_average_return(self):
        """All negative returns (mu <= 0) should indicate no recovery estimate."""
        negative_returns = [-0.01, -0.02, -0.015]
        result = estimate_recovery(negative_returns, 0.05)
        self.assertIsNone(result.expected_recovery_days)
        self.assertIsNone(result.expected_recovery_date)
        self.assertAlmostEqual(result.probability_recover_30d, 0.1)
        self.assertAlmostEqual(result.probability_recover_90d, 0.25)
        self.assertEqual(result.current_drawdown_pct, 0.05)
        self.assertLessEqual(result.avg_daily_return, 0)

    def test_deterministic_simulation(self):
        """
        When the simulated returns are deterministic (sigma=0), the median recovery
        days should equal the ceiling of drawdown / mu.
        """
        # Historical returns with positive mean
        historical_returns = [0.01, 0.02, 0.015]
        drawdown = 0.05  # 5%

        # Compute expected mu
        expected_mu = float(np.mean(historical_returns))
        expected_days = int(np.ceil(drawdown / expected_mu))

        # Mock np.random.normal to return an array filled with expected_mu
        def mock_normal(mu, sigma, size):
            # Ensure mu matches the calculated expected_mu and sigma is ignored
            return np.full(size, expected_mu)

        with patch("numpy.random.normal", side_effect=mock_normal):
            result = estimate_recovery(historical_returns, drawdown)

        self.assertEqual(result.expected_recovery_days, expected_days)
        # The probabilities should be 1.0 because deterministic growth guarantees recovery
        self.assertAlmostEqual(result.probability_recover_30d, 1.0)
        self.assertAlmostEqual(result.probability_recover_90d, 1.0)
        # Verify that the recovery date aligns with the computed days
        self.assertEqual(
            result.expected_recovery_date,
            date.today() + timedelta(days=expected_days),
        )
        self.assertAlmostEqual(result.avg_daily_return, expected_mu)


if __name__ == "__main__":
    unittest.main()