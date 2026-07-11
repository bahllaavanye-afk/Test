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
    """
    Convert a correlation matrix to a distance matrix.

    The distance metric is defined as:
        d = sqrt(0.5 * (1 - rho))

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix where rows and columns correspond to the same assets.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix with zeros on the diagonal.
    """
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)
    return dist


def _get_quasi_diag(link: np.ndarray) -> List[int]:
    """
    Obtain a quasi‑diagonal ordering of assets from a hierarchical linkage matrix.

    The function extracts the leaf order from the dendrogram produced by the
    hierarchical clustering, which is used to sort assets for recursive bisection.

    Parameters
    ----------
    link : np.ndarray
        The linkage matrix returned by :func:`scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    List[int]
        List of asset indices ordered according to the dendrogram leaf order.
    """
    root, _ = to_tree(link, rd=True)
    return leaves_list(link).tolist()


def _get_cluster_var(cov: pd.DataFrame, items: List[int]) -> float:
    """
    Compute the variance of the minimum‑variance portfolio for a sub‑cluster.

    The variance is calculated using the inverse‑variance weighting scheme on the
    sub‑covariance matrix defined by ``items``.

    Parameters
    ----------
    cov : pd.DataFrame
        Full covariance matrix of asset returns.
    items : List[int]
        Indices of assets belonging to the sub‑cluster.

    Returns
    -------
    float
        Portfolio variance for the sub‑cluster. If the cluster contains a single
        asset, the variance is the asset's own variance.
    """
    sub_cov = cov.iloc[items, items].values
    n = len(items)
    if n == 1:
        return float(sub_cov[0, 0])
    inv_var = 1.0 / np.maximum(np.diag(sub_cov), 1e-10)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def _recursive_bisect(cov: pd.DataFrame, sorted_items: List[int]) -> pd.Series:
    """
    Perform recursive bisection to allocate portfolio weights.

    Starting from an equal‑weight vector, the algorithm repeatedly splits the
    sorted list of assets into two halves, computes the variance of each half,
    and adjusts weights proportionally to the inverse of the cluster variances.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix of asset returns.
    sorted_items : List[int]
        Asset indices ordered according to the quasi‑diagonalisation step.

    Returns
    -------
    pd.Series
        Raw (unnormalised) portfolio weights indexed by the provided asset indices.
    """
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

    The optimizer builds a diversified portfolio using hierarchical clustering
    of asset correlations and recursive bisection based on inverse‑variance
    weighting. It falls back to equal weighting when input data is insufficient
    or when an error occurs during computation.
    """

    def compute_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute HRP portfolio weights for a set of assets.

        The method follows the HRP algorithm:
        1. Clean the returns data (drop all‑NaN columns, fill remaining NaNs with 0).
        2. Compute the correlation and covariance matrices.
        3. Transform correlations to a distance matrix and perform Ward linkage.
        4. Derive a quasi‑diagonal ordering of assets.
        5. Allocate weights via recursive bisection.
        6. Re‑index the weights to the original symbols and normalise.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of asset returns where columns are symbols and rows are dates.
            Must contain at least two assets and ten observations to perform HRP.

        Returns
        -------
        pd.Series
            Portfolio weights that sum to 1.0, indexed by the original asset symbols.
            If the input data is insufficient or an error occurs, equal weights are
            returned.
        """
        symbols = list(returns.columns)
        n = len(symbols)

        if n < 2 or len(returns) < 10:
            return pd.Series(1.0 / max(n, 1), index=symbols)

        # Drop columns with all-NaN and fill remaining NaN with 0
        returns_clean = returns.dropna(axis=1, how="all").fillna(0.0)
        if returns_clean.shape[1] < 2:
            return pd.Series(1.0 / max(n, 1), index=symbols)

        symbols_clean = list(returns_clean.columns)

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