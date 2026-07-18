"""
Almgren-Chriss optimal execution trajectory.
Minimizes implementation shortfall by balancing market impact vs timing risk.

Used for orders $5k-$100k. Returns optimal slice schedule.

Reference: Almgren & Chriss (2000) "Optimal execution of portfolio transactions"
"""

import numpy as np
from typing import Dict, Optional, Sequence


class AlmgrenChriss:
    """
    Optimal execution using Almgren‑Chriss (2000) model.

    Parameters
    ----------
    sigma : float
        Daily volatility of the asset (e.g. 0.02 = 2%).
    eta : float, optional
        Temporary impact coefficient (default 2.5e-7).
    gamma : float, optional
        Permanent impact coefficient (default 2.5e-8).
    risk_aversion : float, optional
        Lambda risk‑aversion parameter (default 1e-6).

    Notes
    -----
    The class only provides the mathematical schedule.  In production a
    higher‑level signal layer should decide *when* to apply the schedule
    based on market conditions.  The helper methods ``should_execute`` and
    ``should_exit`` implement lightweight confirmation filters that can be
    integrated into a trading strategy without changing the core execution
    logic.
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
    # Core Almgren‑Chriss model
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
            Number of discrete time slices.

        Returns
        -------
        np.ndarray
            Array of length ``n_slices`` containing the number of shares to
            trade in each slice.
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
            # Near‑zero kappa: fallback to a uniform (TWAP) schedule
            holdings = shares * (1.0 - t / T)
        else:
            holdings = shares * np.sinh(kappa * (T - t)) / denom

        # Trade amounts = negative difference between consecutive holdings
        trades = -np.diff(holdings)
        return trades

    def expected_cost(self, shares: float, T: float, n_slices: int) -> Dict[str, float]:
        """
        Estimate the expected implementation shortfall components.

        Returns
        -------
        dict
            ``temporary_impact``, ``permanent_impact``, ``timing_risk`` and
            ``total`` cost values.
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
    # Strategy‑level helper methods
    # --------------------------------------------------------------------- #
    def _validate_market_data(
        self,
        price_series: Sequence[float],
        spread: float,
        volume: float,
    ) -> None:
        """Validate input market data and raise informative errors."""
        if len(price_series) < 2:
            raise ValueError("price_series must contain at least two price points")
        if spread <= 0:
            raise ValueError(f"spread must be positive, got {spread}")
        if volume <= 0:
            raise ValueError(f"volume must be positive, got {volume}")

    def _recent_volatility(self, price_series: Sequence[float]) -> float:
        """
        Compute the realised volatility of the most recent price window.

        Uses the standard deviation of log‑returns.
        """
        returns = np.diff(np.log(price_series))
        return float(np.std(returns))

    def should_execute(
        self,
        price_series: Sequence[float],
        spread: float,
        volume: float,
        max_spread: float = 0.01,
        min_volume: float = 1e5,
        max_volatility_factor: float = 1.5,
    ) -> bool:
        """
        Confirmation filter before launching an Almgren‑Chriss schedule.

        Entry is allowed only when:
        * Market spread is below ``max_spread`` (tight spread).
        * Available volume exceeds ``min_volume`` (sufficient liquidity).
        * Recent realised volatility is not more than ``max_volatility_factor``
          times the model's daily sigma (avoids extreme volatility spikes).

        Returns
        -------
        bool
            ``True`` if all conditions are satisfied, otherwise ``False``.
        """
        self._validate_market_data(price_series, spread, volume)

        # Tight spread filter
        if spread > max_spread:
            return False

        # Liquidity filter
        if volume < min_volume:
            return False

        # Volatility filter
        recent_vol = self._recent_volatility(price_series)
        if recent_vol > max_volatility_factor * self.sigma:
            return False

        return True

    def should_exit(
        self,
        realized_cost: float,
        cost_threshold: float = 0.02,
    ) -> bool:
        """
        Simple exit filter based on realised implementation shortfall.

        Parameters
        ----------
        realized_cost : float
            The actual cost incurred so far (as a fraction of trade value).
        cost_threshold : float, optional
            Upper bound for acceptable cost; default 2 %.

        Returns
        -------
        bool
            ``True`` if the cost exceeds the threshold and the execution
            should be aborted or re‑routed.
        """
        if realized_cost < 0:
            raise ValueError(f"realized_cost must be non‑negative, got {realized_cost}")
        return realized_cost > cost_threshold

    def get_execution_plan(
        self,
        shares: float,
        T: float,
        n_slices: int,
        price_series: Sequence[float],
        spread: float,
        volume: float,
        **exec_kwargs,
    ) -> Optional[np.ndarray]:
        """
        Produce an execution schedule only when entry filters are satisfied.

        This method encapsulates the typical workflow:
        1. Run ``should_execute`` to verify market conditions.
        2. If the check passes, compute the optimal trajectory.
        3. Return ``None`` when conditions are not met (caller can decide
           to postpone execution).

        Parameters
        ----------
        shares, T, n_slices : see ``optimal_trajectory``.
        price_series, spread, volume : market data for confirmation filters.
        exec_kwargs : additional keyword arguments forwarded to ``should_execute``
                      (e.g., custom ``max_spread`` or ``min_volume``).

        Returns
        -------
        np.ndarray or None
            The trade schedule if conditions are met, otherwise ``None``.
        """
        if self.should_execute(price_series, spread, volume, **exec_kwargs):
            return self.optimal_trajectory(shares, T, n_slices)
        return None

    # --------------------------------------------------------------------- #
    # Dynamic rebalancing (optional advanced feature)
    # --------------------------------------------------------------------- #
    def adjust_trajectory(
        self,
        current_shares: float,
        elapsed_time: float,
        total_time: float,
        remaining_slices: int,
        price_series: Sequence[float],
        spread: float,
        volume: float,
        **adjust_kwargs,
    ) -> np.ndarray:
        """
        Re‑calculate the remaining schedule after a partial execution.

        This method can be called midway through an execution to adapt to
        changing market conditions while preserving the Almgren‑Chriss
        optimality principle.

        Parameters
        ----------
        current_shares : float
            Number of shares still to be liquidated.
        elapsed_time : float
            Minutes already elapsed since the start of execution.
        total_time : float
            Original total execution horizon (minutes).
        remaining_slices : int
            Number of slices left (including the current one).
        price_series, spread, volume : market data for confirmation filters.
        adjust_kwargs : passed to ``should_execute``; if the filter fails,
                       the method falls back to a TWAP schedule for robustness.

        Returns
        -------
        np.ndarray
            Adjusted trade amounts for the remaining slices.
        """
        # Validate that we still have time and slices left
        if remaining_slices <= 0:
            raise ValueError("remaining_slices must be positive")
        if elapsed_time >= total_time:
            raise ValueError("elapsed_time must be less than total_time")

        # If market conditions deteriorated, revert to a simple uniform schedule
        if not self.should_execute(price_series, spread, volume, **adjust_kwargs):
            # Uniform slice (TWAP) for the remaining horizon
            uniform = np.full(remaining_slices, current_shares / remaining_slices)
            return uniform

        # Otherwise compute a fresh Almgren‑Chriss schedule for the residual horizon
        residual_time = total_time - elapsed_time
        return self.optimal_trajectory(current_shares, residual_time, remaining_slices)