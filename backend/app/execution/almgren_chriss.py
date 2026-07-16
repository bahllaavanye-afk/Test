"""
Almgren‑Chriss optimal execution trajectory.

This module implements the Almgren‑Chriss (2000) framework for optimal
execution of a block trade. The model balances market impact against timing
risk to minimise the expected implementation shortfall. It is suitable for
order sizes roughly between $5k and $100k.

Reference
---------
Almgren, R., & Chriss, N. (2000). *Optimal execution of portfolio
transactions*. Journal of Risk, 3(2), 5‑39.
"""

import numpy as np
from typing import Dict, Any


class AlmgrenChriss:
    """
    Optimal execution using the Almgren‑Chriss model.

    Parameters
    ----------
    sigma : float
        Daily volatility of the asset (e.g. 0.02 for 2%).
    eta : float, optional
        Temporary impact coefficient. Default is 2.5e-7.
    gamma : float, optional
        Permanent impact coefficient. Default is 2.5e-8.
    risk_aversion : float, optional
        Risk‑aversion (lambda) parameter. Default is 1e-6.

    Notes
    -----
    The model assumes a linear temporary impact and a linear permanent impact.
    The risk‑aversion parameter controls the trade‑off between impact cost and
    timing risk. Larger values of ``risk_aversion`` lead to more aggressive,
    front‑loaded schedules.

    Example
    -------
    >>> ac = AlmgrenChriss(sigma=0.02)
    >>> schedule = ac.optimal_trajectory(shares=10000, T=30, n_slices=10)
    >>> schedule.shape
    (10,)
    """

    def __init__(
        self,
        sigma: float = 0.02,
        eta: float = 2.5e-7,
        gamma: float = 2.5e-8,
        risk_aversion: float = 1e-6,
    ) -> None:
        """
        Initialise the Almgren‑Chriss model parameters.

        Raises
        ------
        ValueError
            If ``sigma`` or ``eta`` are non‑positive, or if ``risk_aversion``
            is negative.
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
            trade in each slice. The sum of the array equals ``shares``.

        Notes
        -----
        The solution uses the closed‑form sinh‑weighted trajectory derived in
        Almgren & Chriss (2000). When the denominator of the sinh term is
        near‑zero (i.e., very low risk aversion), the method falls back to a
        TWAP‑style uniform slicing.
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
            # Near‑zero kappa: fallback to uniform slicing (TWAP)
            holdings = shares * (1.0 - t / T)
        else:
            holdings = shares * np.sinh(kappa * (T - t)) / denom

        # Trade amounts = negative difference between consecutive holdings
        trades = -np.diff(holdings)
        return trades

    def expected_cost(self, shares: float, T: float, n_slices: int) -> Dict[str, float]:
        """
        Estimate the expected implementation shortfall and its components.

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
            * ``temporary_impact`` – Cost from temporary (transient) price impact.
            * ``permanent_impact`` – Cost from permanent price impact.
            * ``timing_risk`` – Variance cost from price uncertainty over execution.
            * ``total`` – Sum of the three components.
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