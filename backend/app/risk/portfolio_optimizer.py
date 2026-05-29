"""
Portfolio optimization module.

Provides two optimizers:
  HRPOptimizer  — Hierarchical Risk Parity (López de Prado 2016)
  CVaROptimizer — Conditional Value-at-Risk minimization (Rockafellar & Uryasev 2000)

Both use scipy only — no external portfolio library required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from app.risk.hrp import HRPOptimizer  # re-export for convenience

__all__ = ["HRPOptimizer", "CVaROptimizer", "optimize_portfolio"]


class CVaROptimizer:
    """
    Minimize Conditional Value-at-Risk (CVaR / Expected Shortfall) via linear programming.
    Rockafellar & Uryasev (2000) reformulation: linear in (weights, auxiliary vars).

    Usage:
        opt = CVaROptimizer(confidence=0.95)
        weights = opt.compute_weights(returns_df)
    """

    def __init__(self, confidence: float = 0.95):
        if not (0.5 < confidence < 1.0):
            raise ValueError("confidence must be in (0.5, 1.0)")
        self.confidence = confidence

    def compute_weights(self, returns: pd.DataFrame, target_return: float | None = None) -> pd.Series:
        """
        Find portfolio weights that minimise CVaR at the given confidence level.

        Args:
            returns: DataFrame of asset returns (T rows × N assets).
            target_return: optional minimum expected return constraint.

        Returns:
            pd.Series of weights summing to 1, indexed by symbol.
            Falls back to equal weights if optimisation fails.
        """
        symbols = list(returns.columns)
        n = len(symbols)

        if n < 2 or len(returns) < 20:
            return pd.Series(1.0 / max(n, 1), index=symbols)

        returns_clean = returns.dropna(axis=1, how="all").fillna(0.0)
        symbols_clean = list(returns_clean.columns)
        n_clean = len(symbols_clean)
        T = len(returns_clean)
        R = returns_clean.values  # (T, n_clean)

        alpha = self.confidence
        # Decision vars: [w_1..w_n, VaR_threshold, z_1..z_T]
        # Minimize: VaR + 1/((1-alpha)*T) * sum(z_t)
        # s.t.    z_t >= 0
        #         z_t >= -R_t @ w - VaR
        #         sum(w) = 1, w >= 0

        n_vars = n_clean + 1 + T  # weights + VaR scalar + z_t

        # Objective: minimize VaR + 1/((1-alpha)*T) * sum(z)
        c = np.zeros(n_vars)
        c[n_clean] = 1.0                            # VaR coefficient
        c[n_clean + 1:] = 1.0 / ((1.0 - alpha) * T)  # z_t coefficients

        # Inequality constraints: -R_t @ w - VaR - z_t <= 0  (i.e., z_t >= -loss - VaR)
        # Form: A_ub @ x <= b_ub
        # Row t: -R[t] @ w - VaR - z_t <= 0
        A_ub_rows = []
        b_ub_rows = []
        for t in range(T):
            row = np.zeros(n_vars)
            row[:n_clean] = -R[t]     # -R_t @ w
            row[n_clean] = -1.0       # -VaR
            row[n_clean + 1 + t] = -1.0  # -z_t
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)

        # Equality: sum(w) = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_clean] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w >= 0, VaR unconstrained, z >= 0
        bounds = [(0.0, 1.0)] * n_clean + [(None, None)] + [(0.0, None)] * T

        # Optional target return constraint (add as extra equality)
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
            if result.success:
                w = result.x[:n_clean]
                w = np.maximum(w, 0.0)
                total = w.sum()
                if total > 0:
                    w = w / total
                else:
                    w = np.ones(n_clean) / n_clean

                out = pd.Series(0.0, index=symbols)
                for i, sym in enumerate(symbols_clean):
                    out[sym] = float(w[i])
                return out
        except Exception:
            pass

        return pd.Series(1.0 / n, index=symbols)


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "hrp",
    confidence: float = 0.95,
) -> pd.Series:
    """
    Convenience function. method='hrp' or 'cvar'.

    Returns pd.Series of weights indexed by symbol, summing to 1.
    """
    if method == "cvar":
        return CVaROptimizer(confidence=confidence).compute_weights(returns)
    return HRPOptimizer().compute_weights(returns)
