"""
Almgren-Chriss optimal execution trajectory.
Minimizes implementation shortfall by balancing market impact vs timing risk.

Used for orders $5k-$100k. Returns optimal slice schedule.

Reference: Almgren & Chriss (2000) "Optimal execution of portfolio transactions"
"""
import numpy as np


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

    _EPSILON = 1e-12
    _DENOM_TOL = 1e-15

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
        self._validate_inputs(shares, T, n_slices)
        kappa = self._compute_kappa()
        holdings = self._compute_holdings(shares, T, n_slices, kappa)
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

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #

    def _validate_inputs(self, shares: float, T: float, n_slices: int) -> None:
        """Validate inputs common to trajectory calculations."""
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

    def _compute_kappa(self) -> float:
        """Compute the kappa parameter from model coefficients."""
        kappa_sq = (self.lam * self.sigma ** 2) / self.eta
        # Guard against negative or extremely small values
        return np.sqrt(max(kappa_sq, self._EPSILON))

    def _compute_holdings(
        self, shares: float, T: float, n_slices: int, kappa: float
    ) -> np.ndarray:
        """
        Compute the optimal holdings at each time step.

        Returns an array of length n_slices + 1 representing remaining shares
        after each slice (including start and end points).
        """
        t = np.linspace(0, T, n_slices + 1)
        denom = np.sinh(kappa * T)
        if abs(denom) < self._DENOM_TOL:
            # Near-zero kappa: fallback to linear TWAP schedule
            return shares * (1.0 - t / T)
        return shares * np.sinh(kappa * (T - t)) / denom