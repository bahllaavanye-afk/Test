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


def _corr_to_distance(corr: pd.DataFrame) -> np.ndarray:
    """Convert a correlation matrix to a distance matrix using d = sqrt(0.5 * (1 - rho))."""
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)
    return dist


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Return the leaf order of a hierarchical clustering dendrogram (quasi‑diagonalisation)."""
    root, _ = to_tree(link, rd=True)  # noqa: F841  # root is unused but required for to_tree side‑effects
    return leaves_list(link).tolist()


def _get_cluster_var(cov: pd.DataFrame, items: list[int]) -> float:
    """Compute the variance of the minimum‑variance portfolio for a sub‑cluster."""
    sub_cov = cov.iloc[items, items].values
    n = len(items)
    if n == 1:
        return float(sub_cov[0, 0])
    inv_var = 1.0 / np.maximum(np.diag(sub_cov), 1e-10)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def _split_cluster(items: list[int]) -> tuple[list[int], list[int]]:
    """Split a list of item indices into two halves (left, right)."""
    mid = len(items) // 2
    return items[:mid], items[mid:]


def _allocate_between_siblings(
    weights: pd.Series,
    cov: pd.DataFrame,
    left: list[int],
    right: list[int],
) -> None:
    """
    Adjust weights for a pair of sibling clusters based on their variances.

    The allocation factor `alpha` is derived from the inverse‑variance weighting
    described in the HRP paper.
    """
    var_left = _get_cluster_var(cov, left)
    var_right = _get_cluster_var(cov, right)

    denom = max(var_left + var_right, 1e-10)
    alpha = 1.0 - var_left / denom

    weights[left] *= alpha
    weights[right] *= (1.0 - alpha)


def _recursive_bisect(cov: pd.DataFrame, sorted_items: list[int]) -> pd.Series:
    """
    Perform recursive bisection on the ordered list of asset indices.

    The algorithm iteratively splits clusters and allocates weights between the
    resulting sibling clusters using inverse‑variance logic.
    """
    # Initialise equal weight for every asset in the ordered list
    weights = pd.Series(1.0, index=sorted_items)

    # Start with the whole ordered list as the first cluster to bisect
    clusters_to_process: list[list[int]] = [sorted_items]

    while clusters_to_process:
        # Generate next‑level clusters by splitting each current cluster in half
        next_level: list[list[int]] = []
        for cluster in clusters_to_process:
            if len(cluster) > 1:
                left, right = _split_cluster(cluster)
                next_level.extend([left, right])
        # Allocate weights between sibling pairs (left/right) from the previous level
        for i in range(0, len(next_level), 2):
            if i + 1 >= len(next_level):
                break
            _allocate_between_siblings(weights, cov, next_level[i], next_level[i + 1])
        clusters_to_process = next_level

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
        symbols = list(returns.columns)
        n = len(symbols)

        if n < 2 or len(returns) < 10:
            return pd.Series(1.0 / max(n, 1), index=symbols)

        # Drop columns that are entirely NaN and replace remaining NaNs with zeros
        returns_clean = returns.dropna(axis=1, how="all").fillna(0.0)
        if returns_clean.shape[1] < 2:
            return pd.Series(1.0 / max(n, 1), index=symbols)

        symbols_clean = list(returns_clean.columns)
        n_clean = len(symbols_clean)

        try:
            # Correlation matrix clipped to avoid numerical issues
            corr = returns_clean.corr().clip(-0.9999, 0.9999)
            cov = returns_clean.cov()

            # Convert correlation to distance, then to condensed form for clustering
            dist = _corr_to_distance(corr)
            condensed = squareform(dist, checks=False)

            # Hierarchical clustering and ordering
            link = linkage(condensed, method="ward")
            sorted_items = _get_quasi_diag(link)

            # Compute raw weights using recursive bisection
            weights_raw = _recursive_bisect(cov, sorted_items)

            # Map raw weights back to the original symbol order
            result = pd.Series(0.0, index=symbols)
            for idx, sym in enumerate(symbols_clean):
                if idx in weights_raw.index:
                    result[sym] = float(weights_raw[idx])

            # Normalise to ensure weights sum to one
            total = result.sum()
            if total > 0:
                result = result / total
            else:
                result = pd.Series(1.0 / n, index=symbols)

            return result

        except Exception:
            # Fallback to equal weighting on any unexpected error
            return pd.Series(1.0 / n, index=symbols)