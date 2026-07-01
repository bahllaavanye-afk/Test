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

    The implementation follows the linear‑programming reformulation of Rockafellar & Uryasev
    (2000).  The optimisation variables consist of the portfolio weights, a VaR (Value‑at‑Risk)
    scalar, and auxiliary slack variables for each observation.  The optimiser can optionally
    enforce a minimum expected return constraint.

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
        # ----------------------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------------------
        if not isinstance(returns, pd.DataFrame):
            logger.error(
                "CVaROptimizer.compute_weights received non-DataFrame input",
                type=type(returns).__name__,
            )
            raise TypeError("returns must be a pandas DataFrame")

        if returns.empty or returns.columns.empty:
            logger.warning(
                "CVaROptimizer.compute_weights received empty DataFrame, falling back to equal weight"
            )
            # Fallback to equal weighting (will be 0/0 if no columns; handled later)
            return pd.Series(dtype=float)

        symbols = list(returns.columns)
        n = len(symbols)

        # Basic sanity checks – fall back to equal weighting if data are too sparse.
        if n < 2 or len(returns) < 20:
            logger.warning(
                "Insufficient data for CVaROptimizer: assets=%d, observations=%d; using equal weighting",
                n,
                len(returns),
            )
            return pd.Series(1.0 / max(n, 1), index=symbols)

        # Clean data: drop completely empty columns and replace remaining NaNs with zero.
        returns_clean = returns.dropna(axis=1, how="all").fillna(0.0)
        symbols_clean = list(returns_clean.columns)
        n_clean = len(symbols_clean)
        T = len(returns_clean)
        R = returns_clean.values  # shape (T, n_clean)

        alpha = self.confidence

        # Decision variables layout:
        #   [w_1 … w_n, VaR, z_1 … z_T]
        n_vars = n_clean + 1 + T

        # Objective: minimise VaR + (1/((1‑α)·T))·Σ z_t
        c = np.zeros(n_vars)
        c[n_clean] = 1.0                                 # VaR coefficient
        c[n_clean + 1 :] = 1.0 / ((1.0 - alpha) * T)      # z_t coefficients

        # Inequality constraints: -R_t·w - VaR - z_t ≤ 0   (i.e. z_t ≥ –loss – VaR)
        A_ub = np.zeros((T, n_vars))
        b_ub = np.zeros(T)
        for t in range(T):
            A_ub[t, :n_clean] = -R[t]        # -R_t·w
            A_ub[t, n_clean] = -1.0          # -VaR
            A_ub[t, n_clean + 1 + t] = -1.0  # -z_t

        # Equality constraint: Σ w_i = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_clean] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w_i ∈ [0, 1], VaR unrestricted, z_t ≥ 0
        bounds = [(0.0, 1.0)] * n_clean + [(None, None)] + [(0.0, None)] * T

        # Optional expected‑return constraint.
        if target_return is not None:
            mu = returns_clean.mean().values
            ret_row = np.zeros((1, n_vars))
            ret_row[0, :n_clean] = mu
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
        except (ValueError, RuntimeError) as exc:
            logger.error(
                "Linear programming failed in CVaROptimizer.compute_weights",
                error=str(exc),
                exc_info=True,
            )
            # Fallback to equal weighting
            return pd.Series(1.0 / n, index=symbols)

        if not result.success:
            logger.warning(
                "CVaROptimizer.optimize did not converge",
                status=result.status,
                message=result.message,
            )
            return pd.Series(1.0 / n, index=symbols)

        # ----------------------------------------------------------------------
        # Successful optimisation – post‑process weights
        # ----------------------------------------------------------------------
        w = result.x[:n_clean]
        w = np.maximum(w, 0.0)
        total = w.sum()
        w = w / total if total > 0 else np.ones(n_clean) / n_clean

        # Map the cleaned weights back onto the original symbol list.
        out = pd.Series(0.0, index=symbols)
        for i, sym in enumerate(symbols_clean):
            out[sym] = float(w[i])
        return out


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
    if not isinstance(returns, pd.DataFrame):
        logger.error(
            "optimize_portfolio received non-DataFrame input",
            type=type(returns).__name__,
        )
        raise TypeError("returns must be a pandas DataFrame")

    if method == "cvar":
        return CVaROptimizer(confidence=confidence).compute_weights(returns)
    if method == "equal":
        n = len(returns.columns)
        if n == 0:
            logger.warning("optimize_portfolio called with empty DataFrame for equal weighting")
            return pd.Series(dtype=float)
        return pd.Series(1.0 / n, index=returns.columns)
    if method != "hrp":
        logger.error(
            "optimize_portfolio received unknown method",
            method=method,
        )
        raise ValueError(f"Unknown method '{method}'. Choose 'hrp', 'cvar', or 'equal'.")
    return HRPOptimizer().compute_weights(returns)