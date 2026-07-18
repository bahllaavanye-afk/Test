"""
Hierarchical Risk Parity (HRP) portfolio construction.

Reference:
    López de Prado (2016) – “Building Diversified Portfolios that Outperform Out‑of‑Sample”.

Implementation notes
--------------------
* Pure ``scipy``/``numpy`` implementation – no external risk‑parity libraries.
* HRP avoids inverting noisy covariance matrices, generally yielding better out‑of‑sample
  performance compared with classic mean‑variance optimisation.
* The algorithm follows four steps:
    1. Convert the correlation matrix to a distance matrix:
       ``d_ij = sqrt(0.5 * (1 - rho_ij))``.
    2. Perform hierarchical clustering (Ward linkage) on the distance matrix.
    3. Quasi‑diagonalise the clustering dendrogram to obtain an ordering of assets.
    4. Recursively bisect the ordered assets, allocating capital based on inverse‑variance
       within each cluster.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from app.utils.logging import logger
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial.distance import squareform


def _corr_to_distance(corr: pd.DataFrame) -> np.ndarray:
    """
    Convert a correlation matrix to a distance matrix.

    The HRP distance metric is defined as:

        d_ij = sqrt(0.5 * (1 - rho_ij))

    Parameters
    ----------
    corr: pd.DataFrame
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
    Produce a quasi‑diagonal ordering of assets from a linkage matrix.

    The ordering corresponds to the leaf order of the dendrogram produced by
    hierarchical clustering, which groups together assets that are close in the
    correlation‑derived distance space.

    Parameters
    ----------
    link: np.ndarray
        Linkage matrix returned by ``scipy.cluster.hierarchy.linkage``.

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

    The variance is calculated using inverse‑variance weighting within the
    sub‑covariance matrix defined by ``items``.

    Parameters
    ----------
    cov: pd.DataFrame
        Covariance matrix of asset returns.
    items: List[int]
        Indices of assets belonging to the sub‑cluster.

    Returns
    -------
    float
        Portfolio variance for the sub‑cluster.
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
    Perform recursive bisection of assets to allocate weights.

    The algorithm repeatedly splits the ordered list of assets into two halves,
    computes the variance of each half, and allocates capital proportionally to
    the inverse of those variances.

    Parameters
    ----------
    cov: pd.DataFrame
        Covariance matrix of asset returns.
    sorted_items: List[int]
        Asset indices ordered by the quasi‑diagonalisation step.

    Returns
    -------
    pd.Series
        Raw (unnormalised) weights indexed by the original asset indices.
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

    The optimizer builds a diversified portfolio by leveraging hierarchical
    clustering of asset correlations. It is robust to noisy covariance estimates
    and does not require matrix inversion.

    Example
    -------
    >>> optimizer = HRPOptimizer()
    >>> weights = optimizer.compute_weights(returns_df)   # returns a pd.Series indexed by symbol
    """

    def compute_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute HRP portfolio weights for a given set of asset returns.

        The method follows the full HRP pipeline: correlation → distance → clustering
        → quasi‑diagonal ordering → recursive bisection.  It falls back to equal
        weighting when the input data is insufficient or when numerical issues arise.

        Parameters
        ----------
        returns: pd.DataFrame
            DataFrame of asset returns where columns correspond to symbols and rows to
            observation dates.  The DataFrame must contain at least two assets and ten
            observations to produce a meaningful HRP allocation.

        Returns
        -------
        pd.Series
            Portfolio weights that sum to 1.0, indexed by the original symbols.
            If the input data is inadequate or an error occurs, the method returns
            equal weights across the available symbols.
        """
        symbols = list(returns.columns)
        n = len(symbols)

        if n < 2 or len(returns) < 10:
            return pd.Series(1.0 / max(n, 1), index=symbols)

        # Remove assets that are completely NaN and replace remaining NaNs with zero.
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

            # ``sorted_items`` contains indices into ``symbols_clean``.
            weights_raw = _recursive_bisect(cov, sorted_items)

            # Map the raw weights back to the original symbol list.
            result = pd.Series(0.0, index=symbols)
            for idx, sym in enumerate(symbols_clean):
                if idx in weights_raw.index:
                    result[sym] = float(weights_raw[idx])

            # Normalise to ensure the weights sum to one.
            total = result.sum()
            if total > 0:
                result = result / total
            else:
                result = pd.Series(1.0 / n, index=symbols)

            return result

        except Exception as exc:  # noqa: BLE001
            # Log the exception and gracefully degrade to equal weighting.
            logger.warning(
                "HRP optimization failed — falling back to equal weights (%s)", exc
            )
            return pd.Series(1.0 / n, index=symbols)