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

from app.utils.logging import logger
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial.distance import squareform


def _corr_to_distance(corr: pd.DataFrame | None) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5*(1-rho))."""
    if corr is None or corr.empty:
        return np.array([], dtype=float)
    # Ensure numeric values and handle potential NaNs
    corr_vals = corr.values.astype(float)
    dist = np.sqrt(0.5 * (1.0 - corr_vals))
    np.fill_diagonal(dist, 0.0)
    return dist


def _get_quasi_diag(link: np.ndarray | None) -> list[int]:
    """Sort clustered items by the dendrogram leaf order (quasi-diagonalisation)."""
    if link is None or link.size == 0:
        return []
    root, _ = to_tree(link, rd=True)  # root is unused but required for to_tree side‑effects
    return leaves_list(link).tolist()


def _get_cluster_var(cov: pd.DataFrame, items: list[int]) -> float:
    """Minimum-variance portfolio variance for a sub‑cluster."""
    if not items:
        return 0.0
    sub_cov = cov.iloc[items, items].values
    n = len(items)
    if n == 1:
        return float(sub_cov[0, 0])
    # Guard against near‑zero diagonal entries
    inv_var = 1.0 / np.maximum(np.diag(sub_cov), 1e-10)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def _recursive_bisect(cov: pd.DataFrame, sorted_items: list[int]) -> pd.Series:
    """Recursive bisection: split into two halves and allocate by inverse cluster variance."""
    if not sorted_items:
        return pd.Series(dtype=float)
    weights = pd.Series(1.0, index=sorted_items, dtype=float)
    items_to_bisect = [sorted_items]

    while items_to_bisect:
        # Generate new partitions; ensure we only split when length > 1
        items_to_bisect = [
            segment[j:k]
            for segment in items_to_bisect
            for j, k in ((0, len(segment) // 2), (len(segment) // 2, len(segment)))
            if len(segment) > 1
        ]
        # Process pairs of left/right clusters
        for i in range(0, len(items_to_bisect), 2):
            if i + 1 >= len(items_to_bisect):
                break
            left = items_to_bisect[i]
            right = items_to_bisect[i + 1]
            var_left = _get_cluster_var(cov, left)
            var_right = _get_cluster_var(cov, right)
            denominator = max(var_left + var_right, 1e-10)
            alpha = 1.0 - var_left / denominator
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

    def compute_weights(self, returns: pd.DataFrame | None) -> pd.Series:
        """
        Compute HRP weights for a universe of assets.

        Args:
            returns: DataFrame of asset returns, columns = symbols, rows = dates.
                     Must have at least 2 assets and 10 rows.

        Returns:
            pd.Series of portfolio weights summing to 1.0, indexed by symbol.
            Falls back to equal weights if data is insufficient or degenerate.
        """
        if returns is None or returns.empty or returns.shape[1] == 0:
            return pd.Series(dtype=float)

        symbols = list(returns.columns)
        n = len(symbols)

        # Basic sanity checks
        if n < 2 or len(returns) < 10:
            return pd.Series(1.0 / max(n, 1), index=symbols, dtype=float)

        # Drop columns with all-NaN and fill remaining NaN with 0
        returns_clean = returns.dropna(axis=1, how="all").fillna(0.0)
        if returns_clean.shape[1] < 2:
            return pd.Series(1.0 / max(n, 1), index=symbols, dtype=float)

        symbols_clean = list(returns_clean.columns)
        n_clean = len(symbols_clean)

        try:
            corr = returns_clean.corr().clip(-0.9999, 0.9999)
            cov = returns_clean.cov()

            dist = _corr_to_distance(corr)
            if dist.size == 0:
                raise ValueError("Empty distance matrix derived from correlation matrix.")

            condensed = squareform(dist, checks=False)

            link = linkage(condensed, method="ward")
            sorted_items = _get_quasi_diag(link)

            # Guard against empty clustering result
            if not sorted_items:
                raise ValueError("Quasi-diagonalisation produced no ordering.")

            # sorted_items contains indices into symbols_clean
            weights_raw = _recursive_bisect(cov, sorted_items)

            # Re-index back to original symbols
            result = pd.Series(0.0, index=symbols, dtype=float)
            for idx, sym in enumerate(symbols_clean):
                if idx in weights_raw.index:
                    result[sym] = float(weights_raw[idx])

            # Normalise
            total = result.sum()
            if total > 0:
                result = result / total
            else:
                result = pd.Series(1.0 / n, index=symbols, dtype=float)

            return result

        except Exception as exc:  # noqa: BLE001
            # Log the failure and fall back to equal weights.
            logger.warning(
                "HRP optimization failed — falling back to equal weights (%s)", exc
            )
            return pd.Series(1.0 / n, index=symbols, dtype=float)