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
from scipy.cluster.hierarchy import linkage, leaves_list, to_tree
from scipy.spatial.distance import squareform
from typing import List, Tuple


def _corr_to_distance(corr: pd.DataFrame) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5*(1-rho))."""
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)
    return dist


def _get_quasi_diag(link: np.ndarray) -> List[int]:
    """Sort clustered items by the dendrogram leaf order (quasi-diagonalisation)."""
    # `leaves_list` already returns the leaf order for the given linkage matrix.
    return leaves_list(link).tolist()


def _get_cluster_var(cov: pd.DataFrame, items: List[int]) -> float:
    """Minimum-variance portfolio variance for a sub-cluster."""
    sub_cov = cov.iloc[items, items].values
    n = len(items)
    if n == 1:
        return float(sub_cov[0, 0])
    inv_var = 1.0 / np.maximum(np.diag(sub_cov), 1e-10)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def _recursive_bisect(cov: pd.DataFrame, sorted_items: List[int]) -> pd.Series:
    """Recursive bisection: split into two halves and allocate by inverse cluster variance."""
    weights = pd.Series(1.0, index=sorted_items)
    items_to_bisect = [sorted_items]

    while items_to_bisect:
        # Split each cluster into left/right halves
        items_to_bisect = [
            segment[start:end]
            for segment in items_to_bisect
            for start, end in ((0, len(segment) // 2), (len(segment) // 2, len(segment)))
            if len(segment) > 1
        ]

        # Process pairs of adjacent clusters
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
        symbols = list(returns.columns)
        n_assets = len(symbols)

        # Quick sanity checks – fallback to equal weighting if unmet
        if n_assets < 2 or len(returns) < 10:
            return pd.Series(1.0 / max(n_assets, 1), index=symbols)

        clean_returns = self._clean_returns(returns)
        if clean_returns.shape[1] < 2:
            return pd.Series(1.0 / max(n_assets, 1), index=symbols)

        try:
            corr, cov = self._calc_corr_cov(clean_returns)
            sorted_items = self._cluster_and_sort(corr, cov)
            raw_weights = _recursive_bisect(cov, sorted_items)
            result = self._reindex_and_normalise(raw_weights, clean_returns.columns, symbols, n_assets)
            return result
        except Exception:
            # Defensive fallback – any unexpected error yields equal weights
            return pd.Series(1.0 / n_assets, index=symbols)

    def _clean_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Drop entirely NaN columns and fill remaining NaNs with zeros.
        """
        cleaned = returns.dropna(axis=1, how="all").fillna(0.0)
        return cleaned

    def _calc_corr_cov(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute correlation and covariance matrices, clipping correlation extremes.
        """
        corr = returns.corr().clip(-0.9999, 0.9999)
        cov = returns.cov()
        return corr, cov

    def _cluster_and_sort(self, corr: pd.DataFrame, cov: pd.DataFrame) -> List[int]:
        """
        Perform hierarchical clustering on the correlation matrix and return a
        quasi-diagonal ordering of asset indices.
        """
        dist = _corr_to_distance(corr)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="ward")
        return _get_quasi_diag(link)

    def _reindex_and_normalise(
        self,
        raw_weights: pd.Series,
        clean_symbols: pd.Index,
        original_symbols: List[str],
        n_original: int,
    ) -> pd.Series:
        """
        Map weights back to the original symbol list and ensure they sum to one.
        """
        # Map from clean index positions to original symbols
        result = pd.Series(0.0, index=original_symbols)
        for idx, sym in enumerate(clean_symbols):
            if idx in raw_weights.index:
                result[sym] = float(raw_weights[idx])

        total = result.sum()
        if total > 0:
            result = result / total
        else:
            # Degenerate case – revert to equal weighting across original universe
            result = pd.Series(1.0 / n_original, index=original_symbols)
        return result