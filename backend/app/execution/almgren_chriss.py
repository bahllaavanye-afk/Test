"""
Almgren-Chriss optimal execution trajectory.
Minimizes implementation shortfall by balancing market impact vs timing risk.

Used for orders $5k-$100k. Returns optimal slice schedule.

Reference: Almgren & Chriss (2000) "Optimal execution of portfolio transactions"
"""

import numpy as np
from typing import Dict


class AlmgrenChriss:
    """
    Implements the Almgren‑Chriss (2000) optimal execution model.

    The model determines a schedule of trades that minimizes the expected
    implementation shortfall (market impact) plus a risk‑aversion term.

    Parameters
    ----------
    sigma : float
        Daily volatility of the asset (e.g. ``0.02`` for 2%).
    eta : float, optional
        Temporary impact coefficient. Default is ``2.5e-7``.
    gamma : float, optional
        Permanent impact coefficient. Default is ``2.5e-8``.
    risk_aversion : float, optional
        Risk‑aversion parameter (lambda). Default is ``1e-6``.

    Attributes
    ----------
    sigma : float
        Daily volatility.
    eta : float
        Temporary impact coefficient.
    gamma : float
        Permanent impact coefficient.
    lam : float
        Risk‑aversion parameter (alias for ``risk_aversion``).
    """

    def __init__(
        self,
        sigma: float = 0.02,
        eta: float = 2.5e-7,
        gamma: float = 2.5e-8,
        risk_aversion: float = 1e-6,
    ) -> None:
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
        Compute the optimal trade schedule.

        Parameters
        ----------
        shares : float
            Total number of shares to liquidate.
        T : float
            Total execution time in minutes.
        n_slices : int
            Number of time slices (must be positive).

        Returns
        -------
        np.ndarray
            Array of shape ``(n_slices,)`` containing the number of shares to
            trade in each slice.

        Notes
        -----
        The method uses the closed‑form sinh‑weighted trajectory derived in
        Almgren & Chriss (2000). When the ``kappa`` term is effectively zero,
        the schedule falls back to a uniform TWAP slicing.
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

    def expected_cost(self, shares: float, T: float, n_slices: int) -> Dict[str, float]:
        """
        Compute the expected cost breakdown for the optimal schedule.

        Parameters
        ----------
        shares : float
            Total number of shares to liquidate.
        T : float
            Total execution time in minutes.
        n_slices : int
            Number of time slices.

        Returns
        -------
        dict
            Mapping with keys:
            ``temporary_impact`` – cost from temporary (transient) price impact,
            ``permanent_impact`` – cost from permanent price impact,
            ``timing_risk`` – variance cost from price uncertainty over execution,
            ``total`` – sum of the three components.

        Notes
        -----
        The cost components follow the standard Almgren‑Chriss formulation.
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