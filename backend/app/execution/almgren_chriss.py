"""
Almgren-Chriss optimal execution trajectory.
Minimizes implementation shortfall by balancing market impact vs timing risk.

Used for orders $5k-$100k. Returns optimal slice schedule.

Reference: Almgren & Chriss (2000) "Optimal execution of portfolio transactions"
"""
import numpy as np


class AlmgrenChriss:
    """
    Optimal execution using Almgren‑Chriss (2000) model.

    Parameters
    ----------
    sigma : float
        Daily volatility of the asset (e.g., 0.02 for 2%).
    eta : float
        Temporary impact coefficient.
    gamma : float
        Permanent impact coefficient.
    risk_aversion : float
        Lambda parameter governing the trade‑off between cost and risk.

    Usage
    -----
    >>> ac = AlmgrenChriss(sigma=0.02)
    >>> schedule = ac.optimal_trajectory(shares=10000, T=30, n_slices=10)
    >>> # schedule is an array of shares to trade at each time slice
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
            raise ValueError(f"risk_aversion must be non‑negative, got {risk_aversion}")

        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lam = risk_aversion

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def optimal_trajectory(self, shares: float, T: float, n_slices: int) -> np.ndarray:
        """
        Compute the optimal trade schedule.

        Parameters
        ----------
        shares : float
            Total number of shares to liquidate.
        T : float
            Total execution time (minutes).
        n_slices : int
            Number of equal time intervals.

        Returns
        -------
        np.ndarray
            Array of length ``n_slices`` containing the number of shares
            to trade in each interval.
        """
        self._validate_trajectory_inputs(shares, T, n_slices)
        kappa = self._compute_kappa()
        t_grid = np.linspace(0, T, n_slices + 1)
        holdings = self._compute_holdings(shares, T, t_grid, kappa)
        trades = -np.diff(holdings)
        return trades

    def expected_cost(self, shares: float, T: float, n_slices: int) -> dict:
        """
        Compute the expected cost breakdown for the optimal schedule.

        Returns
        -------
        dict
            Keys ``temporary_impact``, ``permanent_impact``, ``timing_risk`` and
            ``total`` (all floats).
        """
        trades = self.optimal_trajectory(shares, T, n_slices)
        tau = T / n_slices

        temp_impact = self.eta * np.sum(trades ** 2) / tau
        perm_impact = 0.5 * self.gamma * shares ** 2
        timing_risk = 0.5 * self.lam * self.sigma ** 2 * np.sum(
            np.cumsum(trades[::-1])[::-1] ** 2 * tau
        )
        total = temp_impact + perm_impact + timing_risk
        return {
            "temporary_impact": float(temp_impact),
            "permanent_impact": float(perm_impact),
            "timing_risk": float(timing_risk),
            "total": float(total),
        }

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _validate_trajectory_inputs(self, shares: float, T: float, n_slices: int) -> None:
        """Validate inputs for ``optimal_trajectory``."""
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")

    def _compute_kappa(self) -> float:
        """
        Compute the ``kappa`` term used in the closed‑form solution.

        Returns
        -------
        float
            Positive square‑root of the risk‑adjusted impact factor.
        """
        kappa_sq = (self.lam * self.sigma ** 2) / self.eta
        # Guard against numerical issues when kappa is near zero.
        kappa_sq = max(kappa_sq, 1e-12)
        return np.sqrt(kappa_sq)

    def _compute_holdings(
        self,
        shares: float,
        T: float,
        t_grid: np.ndarray,
        kappa: float,
    ) -> np.ndarray:
        """
        Compute the optimal holdings at each time point.

        Parameters
        ----------
        shares : float
            Total shares to liquidate.
        T : float
            Total execution horizon.
        t_grid : np.ndarray
            Grid of time points (length ``n_slices + 1``).
        kappa : float
            Pre‑computed kappa value.

        Returns
        -------
        np.ndarray
            Holdings (remaining shares) at each grid point.
        """
        denom = np.sinh(kappa * T)
        if abs(denom) < 1e-15:
            # When kappa is effectively zero, fall back to a uniform (TWAP) schedule.
            return shares * (1.0 - t_grid / T)
        return shares * np.sinh(kappa * (T - t_grid)) / denom