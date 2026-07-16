"""
Almgren-Chriss optimal execution trajectory.
Minimizes implementation shortfall by balancing market impact vs timing risk.

Used for orders $5k-$100k. Returns optimal slice schedule.

Reference: Almgren & Chriss (2000) "Optimal execution of portfolio transactions"
"""
import numpy as np
import unittest


class AlmgrenChriss:
    """
    Optimal execution using Almgren-Chriss (2000) model.

    Parameters:
        sigma: daily volatility of asset (e.g. 0.02 = 2%)
        eta: temporary impact coefficient (default 2.5e-7)
        gamma: permanent impact coefficient (default 2.5e-8)
        risk_aversion: lambda parameter (default 1e-6)

    Usage:
        ac = AlmgrenChriss(sigma=0.02)
        schedule = ac.optimal_trajectory(shares=10000, T=30, n_slices=10)
        # Returns array of shares to trade at each time slice
    """

    def __init__(
        self,
        sigma: float = 0.02,
        eta: float = 2.5e-7,
        gamma: float = 2.5e-8,
        risk_aversion: float = 1e-6,
    ):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        if risk_aversion < 0:
            raise ValueError(f"risk_aversion must be non-negative, got {risk_aversion}")
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lam = risk_aversion

    def optimal_trajectory(self, shares: float, T: float, n_slices: int) -> np.ndarray:
        """
        Returns array of shape (n_slices,) with shares to trade per slice.

        T is total execution time in minutes.
        Uses sinh-weighted trajectory (Almgren-Chriss closed-form solution).

        The trajectory minimises E[cost] + lambda * Var[cost] subject to
        liquidating all `shares` within time T.
        """
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        kappa_sq = (self.lam * self.sigma ** 2) / self.eta
        kappa = np.sqrt(max(kappa_sq, 1e-12))
        t = np.linspace(0, T, n_slices + 1)

        # Optimal holdings at each time step
        denom = np.sinh(kappa * T)
        if abs(denom) < 1e-15:
            # Near-zero kappa: TWAP fallback (uniform slicing)
            holdings = shares * (1.0 - t / T)
        else:
            holdings = shares * np.sinh(kappa * (T - t)) / denom

        # Trade amounts = negative difference between consecutive holdings
        trades = -np.diff(holdings)
        return trades

    def expected_cost(self, shares: float, T: float, n_slices: int) -> dict:
        """
        Returns expected market impact cost breakdown.

        Keys:
            temporary_impact: cost from temporary (transient) price impact
            permanent_impact: cost from permanent price impact
            timing_risk: variance cost from price uncertainty over execution
            total: sum of all three components
        """
        trades = self.optimal_trajectory(shares, T, n_slices)
        tau = T / n_slices

        temp_impact = self.eta * np.sum(trades ** 2) / tau
        perm_impact = 0.5 * self.gamma * shares ** 2
        timing_risk = 0.5 * self.lam * self.sigma ** 2 * np.sum(
            np.cumsum(trades[::-1])[::-1] ** 2 * tau
        )
        return {
            "temporary_impact": float(temp_impact),
            "permanent_impact": float(perm_impact),
            "timing_risk": float(timing_risk),
            "total": float(temp_impact + perm_impact + timing_risk),
        }


class TestAlmgrenChrissEdgeCases(unittest.TestCase):
    """Unit tests targeting boundary and edge conditions."""

    def test_single_slice_liquidation(self):
        """n_slices=1 should liquidate the entire position in one trade."""
        ac = AlmgrenChriss()
        shares = 1000.0
        T = 30.0
        trades = ac.optimal_trajectory(shares=shares, T=T, n_slices=1)
        self.assertEqual(trades.shape, (1,))
        self.assertAlmostEqual(trades.sum(), shares, places=10)

    def test_zero_risk_aversion_uniform_slicing(self):
        """When risk aversion (lambda) is zero, trajectory falls back to uniform TWAP."""
        ac = AlmgrenChriss(risk_aversion=0.0)
        shares = 1200.0
        T = 10.0
        n_slices = 4
        trades = ac.optimal_trajectory(shares=shares, T=T, n_slices=n_slices)
        expected = np.full(n_slices, shares / n_slices)
        np.testing.assert_allclose(trades, expected, rtol=1e-12)

    def test_zero_shares_returns_zero_trades(self):
        """A zero share order should result in zero trade amounts regardless of other parameters."""
        ac = AlmgrenChriss()
        trades = ac.optimal_trajectory(shares=0.0, T=20.0, n_slices=5)
        self.assertTrue(np.allclose(trades, 0.0))

    def test_invalid_parameters_raise(self):
        """Negative sigma, eta, or non‑positive n_slices/T should raise ValueError."""
        with self.assertRaises(ValueError):
            AlmgrenChriss(sigma=-0.01)
        with self.assertRaises(ValueError):
            AlmgrenChriss(eta=-1e-7)
        ac = AlmgrenChriss()
        with self.assertRaises(ValueError):
            ac.optimal_trajectory(shares=1000, T=10, n_slices=0)
        with self.assertRaises(ValueError):
            ac.optimal_trajectory(shares=1000, T=-5, n_slices=5)


if __name__ == "__main__":
    unittest.main()