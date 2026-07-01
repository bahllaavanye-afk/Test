"""
Portfolio optimization module.

Provides two optimizers:
  - HRPOptimizer — Hierarchical Risk Parity (López de Prado 2016)
  - CVaROptimizer — Conditional Value‑at‑Risk minimisation (Rockafellar & Uryasev 2000)

Both optimizers rely only on SciPy and pandas; no external portfolio library is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from scipy.optimize import linprog
from typing import Optional

from app.risk.hrp import HRPOptimizer  # re-export for convenience

logger = structlog.get_logger()

__all__ = ["HRPOptimizer", "CVaROptimizer", "optimize_portfolio"]


class CVaROptimizer:
    """
    Optimiser that minimises Conditional Value‑at‑Risk (CVaR, also known as Expected Shortfall).

    The implementation follows the linear‑programming reformulation of Rockafellar &
    Uryasev (2000).  The optimisation variables consist of the portfolio weights, a VaR
    (Value‑at‑Risk) scalar, and auxiliary slack variables for each observation.  The optimiser
    can optionally enforce a minimum expected return constraint.

    Example
    -------
    >>> opt = CVaROptimizer(confidence=0.95)
    >>> weights = opt.compute_weights(returns_df)
    """

    def __init__(self, confidence: float = 0.95) -> None:
        """
        Parameters
        ----------
        confidence: float, default 0.95
            Confidence level for CVaR (must be between 0.5 and 1.0 exclusive).
        """
        if not (0.5 < confidence < 1.0):
            raise ValueError("confidence must be in (0.5, 1.0)")
        self.confidence = confidence

    def _prepare_data(self, returns: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
        """
        Clean the input DataFrame and return a tuple with the cleaned data,
        the list of symbols and the original asset count.

        Returns
        -------
        returns_clean : pd.DataFrame
            DataFrame with NaNs filled with 0 and completely empty columns removed.
        symbols : list[str]
            Original symbol order.
        n_original : int
            Number of assets before cleaning.
        """
        symbols = list(returns.columns)
        n_original = len(symbols)

        # Drop columns that are entirely NaN; fill remaining NaNs with zero.
        returns_clean = returns.dropna(axis=1, how="all").fillna(0.0)

        return returns_clean, symbols, n_original

    def compute_weights(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
    ) -> pd.Series:
        """
        Compute portfolio weights that minimise CVaR at the configured confidence level.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns with shape (T, N) where T is the number of observations and N
            is the number of assets. Columns are asset symbols.
        target_return : float | None, optional
            If provided, an equality constraint ``w·μ = target_return`` is added,
            where μ is the vector of mean returns.

        Returns
        -------
        pd.Series
            Portfolio weights indexed by the original asset symbols.  The weights sum to
            one.  If the optimisation fails or the input data are insufficient, equal
            weighting is returned as a safe fallback.
        """
        returns_clean, symbols, n_original = self._prepare_data(returns)
        symbols_clean = list(returns_clean.columns)
        n = len(symbols_clean)
        T = len(returns_clean)

        # Basic sanity checks – fall back to equal weighting if data are too sparse.
        if n < 2 or T < 20:
            logger.info(
                "Insufficient data for CVaR optimisation; falling back to equal weighting",
                assets=n,
                observations=T,
            )
            return pd.Series(1.0 / max(n_original, 1), index=symbols)

        R = returns_clean.values  # shape (T, n)

        alpha = self.confidence

        # Decision variables layout:
        #   [w_1 … w_n, VaR, z_1 … z_T]
        n_vars = n + 1 + T

        # Objective: minimise VaR + (1/((1‑α)·T))·Σ z_t
        c = np.zeros(n_vars)
        c[n] = 1.0                                 # VaR coefficient
        c[n + 1 :] = 1.0 / ((1.0 - alpha) * T)      # z_t coefficients

        # Inequality constraints: -R_t·w - VaR - z_t ≤ 0   (i.e. z_t ≥ –loss – VaR)
        # Vectorised construction for speed and clarity.
        A_ub = np.hstack(
            [
                -R,                                 # -R_t·w
                -np.ones((T, 1)),                  # -VaR
                -np.eye(T),                         # -z_t (each row only touches its own slack)
            ]
        )
        b_ub = np.zeros(T)

        # Equality constraint: Σ w_i = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w_i ∈ [0, 1], VaR unrestricted, z_t ≥ 0
        bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * T

        # Optional expected‑return constraint.
        if target_return is not None:
            mu = returns_clean.mean().values
            ret_row = np.zeros((1, n_vars))
            ret_row[0, :n] = mu
            A_eq = np.vstack([A_eq, ret_row])
            b_eq = np.append(b_eq, target_return)

        try:
            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            if not result.success:
                raise RuntimeError(f"Linprog failed: {result.message}")

            # Extract and normalise weights.
            w_raw = result.x[:n]
            w_raw = np.clip(w_raw, 0.0, None)  # enforce non‑negativity
            total = w_raw.sum()
            if total <= 0:
                logger.warning("Optimiser returned non‑positive total weight; using equal weights")
                w = np.full(n, 1.0 / n)
            else:
                w = w_raw / total

            # Map the cleaned weights back onto the original symbol list.
            out = pd.Series(0.0, index=symbols)
            for i, sym in enumerate(symbols_clean):
                out[sym] = float(w[i])
            return out

        except Exception as exc:  # pragma: no cover
            logger.warning(
                "CVaROptimizer.compute_weights failed, falling back to equal weight",
                error=str(exc),
            )
            # Fallback: equal weighting across the original symbols.
            return pd.Series(1.0 / max(n_original, 1), index=symbols)


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "hrp",
    confidence: float = 0.95,
) -> pd.Series:
    """
    Convenience wrapper that selects a portfolio optimisation method.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series, columns correspond to symbols.
    method : str, default "hrp"
        One of ``"hrp"``, ``"cvar"``, or ``"equal"``.
    confidence : float, default 0.95
        Confidence level used by the CVaR optimiser; ignored for other methods.

    Returns
    -------
    pd.Series
        Portfolio weights indexed by symbol and summing to one.
    """
    method = method.lower()
    if method == "cvar":
        return CVaROptimizer(confidence=confidence).compute_weights(returns)
    if method == "equal":
        n = len(returns.columns)
        if n == 0:
            raise ValueError("Cannot compute equal weights for an empty returns DataFrame")
        return pd.Series(1.0 / n, index=returns.columns)
    if method != "hrp":
        raise ValueError(f"Unknown method '{method}'. Choose 'hrp', 'cvar', or 'equal'.")
    return HRPOptimizer().compute_weights(returns)