"""
Almgren-Chriss optimal execution trajectory.
Minimizes implementation shortfall by balancing market impact vs timing risk.

Used for orders $5k-$100k. Returns optimal slice schedule.

Reference: Almgren & Chriss (2000) "Optimal execution of portfolio transactions"
"""
import numpy as np


class AlmgrenChriss:
    """
    Optimal execution using the Almgren‑Chriss (2000) model.

    Parameters
    ----------
    sigma : float
        Daily volatility of the asset (e.g. 0.02 for 2 %).
    eta : float, optional
        Temporary impact coefficient (default 2.5e‑7).
    gamma : float, optional
        Permanent impact coefficient (default 2.5e‑8).
    risk_aversion : float, optional
        λ risk‑aversion parameter (default 1e‑6).

    Notes
    -----
    The model assumes a linear temporary impact and a linear permanent impact.
    It produces a deterministic schedule that minimises the expected
    implementation shortfall plus a penalty on variance (timing risk).
    """

    _MIN_SHARES = 5_000
    _MAX_SHARES = 100_000
    _KAPPA_EPS = 1e-12
    _DENOM_EPS = 1e-15

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
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if risk_aversion < 0:
            raise ValueError(f"risk_aversion must be non‑negative, got {risk_aversion}")

        self.sigma = float(sigma)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.lam = float(risk_aversion)

    # --------------------------------------------------------------------- #
    # Core model utilities
    # --------------------------------------------------------------------- #
    def _kappa(self) -> float:
        """Compute the κ parameter with numerical safeguards."""
        kappa_sq = (self.lam * self.sigma ** 2) / self.eta
        # Guard against negative or extremely small values caused by rounding
        kappa_sq = max(kappa_sq, self._KAPPA_EPS)
        return np.sqrt(kappa_sq)

    # --------------------------------------------------------------------- #
    # Public interface
    # --------------------------------------------------------------------- #
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
            Number of discrete time slices (must be > 0).

        Returns
        -------
        np.ndarray
            Array of shape ``(n_slices,)`` containing the number of shares to
            trade in each slice. The array is of dtype ``float``.
        """
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        kappa = self._kappa()
        t = np.linspace(0, T, n_slices + 1, dtype=np.float64)

        denom = np.sinh(kappa * T)
        if abs(denom) < self._DENOM_EPS:
            # Near‑zero κ: fallback to TWAP (uniform slicing)
            holdings = shares * (1.0 - t / T)
        else:
            holdings = shares * np.sinh(kappa * (T - t)) / denom

        trades = -np.diff(holdings).astype(np.float64)
        return trades

    def expected_cost(self, shares: float, T: float, n_slices: int) -> dict:
        """
        Estimate the expected implementation shortfall cost breakdown.

        Returns
        -------
        dict
            Keys:
            - ``temporary_impact`` – cost from temporary (transient) price impact
            - ``permanent_impact`` – cost from permanent price impact
            - ``timing_risk`` – variance cost from price uncertainty over execution
            - ``total`` – sum of the three components
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
    # Strategy‑level helpers (entry/exit logic)
    # --------------------------------------------------------------------- #
    def should_execute(self, shares: float, market_volatility: float) -> bool:
        """
        Determine whether an execution order should be launched.

        Entry is allowed only when:
        * The order size is within the model's calibrated range.
        * The current market volatility is not excessively higher than the
          model's reference volatility (``sigma``).

        Parameters
        ----------
        shares : float
            Desired order size.
        market_volatility : float
            Real‑time volatility estimate (same units as ``sigma``).

        Returns
        -------
        bool
            ``True`` if the order satisfies the entry filters, otherwise ``False``.
        """
        if not (self._MIN_SHARES <= shares <= self._MAX_SHARES):
            return False
        # Allow execution if volatility is within 150 % of the calibrated sigma
        return market_volatility <= 1.5 * self.sigma

    def confirm_execution(self, shares: float, T: float, n_slices: int, max_impact: float) -> bool:
        """
        Confirmation filter applied after the initial entry check.

        The order is confirmed if the estimated total implementation shortfall
        per share does not exceed ``max_impact`` (expressed in price units).

        Parameters
        ----------
        shares : float
            Order size.
        T : float
            Planned execution horizon (minutes).
        n_slices : int
            Number of slices for the schedule.
        max_impact : float
            Maximum acceptable cost per share.

        Returns
        -------
        bool
            ``True`` if the expected cost per share is within the tolerance.
        """
        cost = self.expected_cost(shares, T, n_slices)["total"]
        return (cost / shares) <= max_impact

    def should_exit(self, remaining_shares: float, time_elapsed: float, T: float) -> bool:
        """
        Simple exit logic for a partially‑filled order.

        Exit when either:
        * Very few shares remain (≤ 1 % of the original order), or
        * The remaining execution window is less than 5 % of the original horizon.

        Parameters
        ----------
        remaining_shares : float
            Shares still to be executed.
        time_elapsed : float
            Minutes elapsed since the start of the execution.
        T : float
            Original total execution time (minutes).

        Returns
        -------
        bool
            ``True`` if the order should be terminated early.
        """
        if remaining_shares <= 0:
            return True
        # Fraction of original order – we assume the original size is known via context
        # For a generic check we compare against a small absolute threshold.
        if remaining_shares < 0.01 * self._MAX_SHARES:
            return True
        remaining_time = max(T - time_elapsed, 0.0)
        return remaining_time / T < 0.05