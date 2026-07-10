"""
Hierarchical Risk Parity (HRP) portfolio construction.
López de Prado (2016) — "Building Diversified Portfolios that Outperform Out-of-Sample".

Pure scipy/numpy implementation — no riskfolio dependency.
HRP avoids inverting noisy covariance matrices, giving better OOS performance than MVO.

Steps:
  1. Compute correlation-based distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))
  2. Hierarchical clustering (Ward linkage on distance matrix)
  3. Quasi-diagonalisation: sort assets by cluster proximity
  4. Recursive bisection: allocate based on inverse-variance within each cluster
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial.distance import squareform
from typing import List


def _corr_to_distance(corr: pd.DataFrame) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5*(1-rho))."""
    if not isinstance(corr, pd.DataFrame):
        raise ValueError("corr must be a pandas DataFrame.")
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square matrix.")
    if corr.empty:
        raise ValueError("corr must not be empty.")
    if not ((corr.values >= -1.0).all() and (corr.values <= 1.0).all()):
        raise ValueError("corr values must lie in the range [-1, 1].")
    # Ensure symmetry
    if not np.allclose(corr.values, corr.values.T, atol=1e-8):
        raise ValueError("corr matrix must be symmetric.")
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)
    return dist


def _get_quasi_diag(link: np.ndarray) -> List[int]:
    """Sort clustered items by the dendrogram leaf order (quasi-diagonalisation)."""
    if not isinstance(link, np.ndarray):
        raise ValueError("link must be a numpy ndarray.")
    if link.ndim != 2 or link.shape[1] != 4:
        raise ValueError("link must have shape (n-1, 4) as produced by scipy linkage.")
    root, _ = to_tree(link, rd=True)
    return leaves_list(link).tolist()


def _get_cluster_var(cov: pd.DataFrame, items: List[int]) -> float:
    """Minimum-variance portfolio variance for a sub‑cluster."""
    if not isinstance(cov, pd.DataFrame):
        raise ValueError("cov must be a pandas DataFrame.")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix.")
    if not items:
        raise ValueError("items list must not be empty.")
    if any(not isinstance(i, int) for i in items):
        raise ValueError("items must be a list of integer indices.")
    max_index = cov.shape[0] - 1
    if any(i < 0 or i > max_index for i in items):
        raise ValueError("item indices are out of bounds for the covariance matrix.")
    sub_cov = cov.iloc[items, items].values
    n = len(items)
    if n == 1:
        return float(sub_cov[0, 0])
    inv_var = 1.0 / np.maximum(np.diag(sub_cov), 1e-10)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def _recursive_bisect(cov: pd.DataFrame, sorted_items: List[int]) -> pd.Series:
    """Recursive bisection: split into two halves and allocate by inverse cluster variance."""
    if not isinstance(cov, pd.DataFrame):
        raise ValueError("cov must be a pandas DataFrame.")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix.")
    if not sorted_items:
        raise ValueError("sorted_items must not be empty.")
    if any(not isinstance(i, int) for i in sorted_items):
        raise ValueError("sorted_items must be a list of integer indices.")
    max_index = cov.shape[0] - 1
    if any(i < 0 or i > max_index for i in sorted_items):
        raise ValueError("sorted_items contain indices out of bounds for the covariance matrix.")
    weights = pd.Series(1.0, index=sorted_items)
    items_to_bisect = [sorted_items]

    while items_to_bisect:
        items_to_bisect = [
            i[j:k]
            for i in items_to_bisect
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(items_to_bisect), 2):
            if i + 1 >= len(items_to_bisect):
                break
            left = items_to_bisect[i]
            right = items_to_bisect[i + 1]
            var_left = _get_cluster_var(cov, left)
            var_right = _get_cluster_var(cov, right)
            alpha = 1.0 - var_left / max(var_left + var_right, 1e-10)
            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)

    return weights


class HRPOptimizer:
    """
    Hierarchical Risk Parity portfolio optimizer.

    Usage:
        hrp = HRPOptimizer()
        weights = hrp.compute_weights(returns_df)  # returns pd.Series indexed by symbol
    """

    def compute_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute HRP weights for a universe of assets.

        Args:
            returns: DataFrame of asset returns, columns = symbols, rows = dates.
                     Must have at least 2 assets and 10 rows.

        Returns:
            pd.Series of portfolio weights summing to 1.0, indexed by symbol.
            Falls back to equal weights if data is insufficient or degenerate.
        """
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("returns must be a pandas DataFrame.")
        if returns.empty:
            raise ValueError("returns DataFrame must not be empty.")
        if returns.shape[1] < 2:
            raise ValueError("returns must contain at least 2 assets (columns).")
        if returns.shape[0] < 10:
            raise ValueError("returns must contain at least 10 observations (rows).")

        symbols = list(returns.columns)
        n = len(symbols)

        # Drop columns with all-NaN and fill remaining NaN with 0
        returns_clean = returns.dropna(axis=1, how="all").fillna(0.0)
        if returns_clean.shape[1] < 2:
            return pd.Series(1.0 / max(n, 1), index=symbols)

        symbols_clean = list(returns_clean.columns)
        n_clean = len(symbols_clean)

        try:
            corr = returns_clean.corr().clip(-0.9999, 0.9999)
            cov = returns_clean.cov()

            dist = _corr_to_distance(corr)
            condensed = squareform(dist, checks=False)

            link = linkage(condensed, method="ward")
            sorted_items = _get_quasi_diag(link)

            # sorted_items contains indices into symbols_clean
            weights_raw = _recursive_bisect(cov, sorted_items)

            # Re-index back to original symbols
            result = pd.Series(0.0, index=symbols)
            for idx, sym in enumerate(symbols_clean):
                if idx in weights_raw.index:
                    result[sym] = float(weights_raw[idx])

            # Normalise
            total = result.sum()
            if total > 0:
                result = result / total
            else:
                result = pd.Series(1.0 / n, index=symbols)

            return result

        except Exception:
            return pd.Series(1.0 / n, index=symbols)