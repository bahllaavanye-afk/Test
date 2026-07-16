"""
Almgren–Chriss optimal execution trajectory.

This module provides a simple implementation of the Almgren‑Chriss (2000) model
for optimal execution. It computes a schedule that minimizes the expected
implementation shortfall while balancing market impact against timing risk.
The model is suitable for order sizes roughly between $5k and $100k.
"""

import numpy as np
from typing import Dict


class AlmgrenChriss:
    """
    Optimal execution using the Almgren‑Chriss (2000) model.

    Parameters
    ----------
    sigma : float
        Daily volatility of the asset (e.g., 0.02 for 2%).
    eta : float, optional
        Temporary impact coefficient. Default is 2.5e-7.
    gamma : float, optional
        Permanent impact coefficient. Default is 2.5e-8.
    risk_aversion : float, optional
        Risk‑aversion (lambda) parameter. Default is 1e-6.

    Notes
    -----
    The model assumes a continuous‑time setting and derives a closed‑form
    sinh‑weighted trajectory for the optimal trade schedule.
    """

    def __init__(
        self,
        sigma: float = 0.02,
        eta: float = 2.5e-7,
        gamma: float = 2.5e-8,
        risk_aversion: float = 1e-6,
    ) -> None:
        """
        Initialise the Almgren‑Chriss model with market and risk parameters.

        Raises
        ------
        ValueError
            If any of the required parameters are non‑positive where positivity
            is required.
        """
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
            Number of discrete time slices (must be positive).

        Returns
        -------
        np.ndarray
            Array of shape ``(n_slices,)`` containing the number of shares to
            trade in each slice.

        Notes
        -----
        The closed‑form solution uses a sinh‑weighted trajectory. When the
        denominator becomes near‑zero, the method falls back to a uniform (TWAP)
        schedule.
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
            # Near‑zero kappa: TWAP fallback (uniform slicing)
            holdings = shares * (1.0 - t / T)
        else:
            holdings = shares * np.sinh(kappa * (T - t)) / denom

        # Trade amounts = negative difference between consecutive holdings
        trades = -np.diff(holdings)
        return trades

    def expected_cost(self, shares: float, T: float, n_slices: int) -> Dict[str, float]:
        """
        Estimate the expected implementation shortfall cost breakdown.

        Parameters
        ----------
        shares : float
            Total number of shares to liquidate.
        T : float
            Total execution time in minutes.
        n_slices : int
            Number of discrete time slices.

        Returns
        -------
        dict
            Mapping with the following keys (all values are ``float``):
            - ``temporary_impact``: Cost from temporary (transient) price impact.
            - ``permanent_impact``: Cost from permanent price impact.
            - ``timing_risk``: Variance cost from price uncertainty over execution.
            - ``total``: Sum of the three components.
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