"""
Almgren-Chriss optimal execution trajectory.
Minimizes implementation shortfall by balancing market impact vs timing risk.

Used for orders $5k-$100k. Returns optimal slice schedule.

Reference: Almgren & Chriss (2000) "Optimal execution of portfolio transactions"
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    The model produces a schedule that minimizes
    ``E[cost] + lambda * Var[cost]`` subject to fully liquidating the
    position within the execution horizon ``T``.
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
        if gamma < 0:
            raise ValueError(f"gamma must be non‑negative, got {gamma}")
        if risk_aversion < 0:
            raise ValueError(f"risk_aversion must be non‑negative, got {risk_aversion}")

        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lam = risk_aversion

    def _validate_execution_params(
        self,
        shares: float,
        T: float,
        n_slices: int,
        min_trade_size: float,
    ) -> None:
        """Internal sanity checks for execution parameters."""
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if n_slices <= 0:
            raise ValueError(f"n_slices must be positive, got {n_slices}")
        if min_trade_size < 0:
            raise ValueError(f"min_trade_size must be non‑negative, got {min_trade_size}")

    def optimal_trajectory(
        self,
        shares: float,
        T: float,
        n_slices: int,
        *,
        min_trade_size: float = 0.0,
        avg_daily_volume: Optional[float] = None,
        max_participation: float = 0.2,
    ) -> np.ndarray:
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
        min_trade_size : float, optional
            Minimum allowable trade size per slice (default 0, i.e., no restriction).
        avg_daily_volume : float, optional
            Estimated average daily volume for the instrument. If provided, the schedule
            is filtered to respect ``max_participation`` of this volume per slice.
        max_participation : float, optional
            Maximum fraction of ``avg_daily_volume`` that can be traded in a single
            slice (default 0.2 = 20%). Ignored if ``avg_daily_volume`` is ``None``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_slices,)`` with shares to trade per slice.
        """
        self._validate_execution_params(shares, T, n_slices, min_trade_size)

        # Compute the classic Almgren‑Chriss sinh‑weighted trajectory
        kappa_sq = (self.lam * self.sigma ** 2) / self.eta
        kappa = np.sqrt(max(kappa_sq, 1e-12))
        t = np.linspace(0, T, n_slices + 1)

        denom = np.sinh(kappa * T)
        if abs(denom) < 1e-15:
            # Near‑zero kappa: fall back to TWAP (uniform slicing)
            holdings = shares * (1.0 - t / T)
        else:
            holdings = shares * np.sinh(kappa * (T - t)) / denom

        trades = -np.diff(holdings)  # Positive values represent executed shares

        # --- Confirmation filters -------------------------------------------------
        # 1. Minimum trade size enforcement
        if min_trade_size > 0:
            # Round up to the nearest multiple of min_trade_size while preserving total shares
            rounded = np.ceil(trades / min_trade_size) * min_trade_size
            diff = rounded.sum() - trades.sum()
            if diff > 0:
                # Reduce the last slice to keep the total unchanged
                rounded[-1] -= diff
                if rounded[-1] < min_trade_size:
                    logger.warning(
                        "Rounding caused last slice to fall below min_trade_size; "
                        "adjusting to min_trade_size and redistributing remainder."
                    )
                    rounded[-1] = min_trade_size
            trades = rounded

        # 2. Participation rate cap
        if avg_daily_volume is not None:
            if avg_daily_volume <= 0:
                raise ValueError(f"avg_daily_volume must be positive, got {avg_daily_volume}")
            max_slice_volume = max_participation * avg_daily_volume * (T / (24 * 60))  # volume per minute * minutes per slice
            slice_volume = trades / (T / n_slices)  # shares per minute in each slice
            over_limit = slice_volume > max_slice_volume
            if np.any(over_limit):
                logger.info(
                    "Applying participation‑rate filter: %d slice(s) exceed max participation.",
                    int(over_limit.sum()),
                )
                # Scale down offending slices proportionally and redistribute excess
                excess = slice_volume[over_limit] - max_slice_volume
                slice_volume[over_limit] = max_slice_volume
                total_excess = np.sum(excess) * (T / n_slices)
                # Redistribute excess uniformly across non‑offending slices
                non_offending = ~over_limit
                if np.any(non_offending):
                    slice_volume[non_offending] += total_excess / np.sum(non_offending)
                else:
                    logger.warning("All slices exceed participation limit; scaling uniformly.")
                    slice_volume = slice_volume * (max_slice_volume / slice_volume.max())
                trades = slice_volume * (T / n_slices)

        # Ensure numerical stability: small negative values can appear due to rounding
        trades = np.where(trades < 0, 0.0, trades)

        # Final sanity check: sum must equal original shares within tolerance
        total_traded = trades.sum()
        if not np.isclose(total_traded, shares, rtol=1e-6, atol=1e-6):
            logger.warning(
                "Total traded shares (%.6f) differ from target (%.6f); adjusting last slice.",
                total_traded,
                shares,
            )
            trades[-1] += shares - total_traded

        return trades

    def expected_cost(self, shares: float, T: float, n_slices: int) -> dict:
        """
        Compute the expected market‑impact cost breakdown for a given schedule.

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
            Dictionary with keys ``temporary_impact``, ``permanent_impact``,
            ``timing_risk`` and ``total``.
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

    def update_schedule(
        self,
        executed_trades: Tuple[np.ndarray, float],
        remaining_shares: float,
        T_remaining: float,
        n_slices_remaining: int,
        *,
        min_trade_size: float = 0.0,
        avg_daily_volume: Optional[float] = None,
        max_participation: float = 0.2,
    ) -> np.ndarray:
        """
        Re‑calculate the execution schedule after a partial fill.

        Parameters
        ----------
        executed_trades : tuple(np.ndarray, float)
            The first element is an array of trades already executed;
            the second element is the elapsed time (minutes) since start.
        remaining_shares : float
            Shares still to be liquidated.
        T_remaining : float
            Remaining execution time (minutes).
        n_slices_remaining : int
            Number of slices left for the remaining horizon.
        min_trade_size, avg_daily_volume, max_participation :
            Same semantics as :meth:`optimal_trajectory`.

        Returns
        -------
        np.ndarray
            Updated trade schedule for the remaining horizon.
        """
        _, elapsed = executed_trades
        if elapsed < 0 or elapsed > T_remaining + elapsed:
            raise ValueError("Elapsed time is out of bounds.")
        return self.optimal_trajectory(
            remaining_shares,
            T_remaining,
            n_slices_remaining,
            min_trade_size=min_trade_size,
            avg_daily_volume=avg_daily_volume,
            max_participation=max_participation,
        )