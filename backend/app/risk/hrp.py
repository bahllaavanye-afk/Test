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


def _corr_to_distance(corr: pd.DataFrame) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5*(1-rho))."""
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)
    return dist


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Sort clustered items by the dendrogram leaf order (quasi-diagonalisation)."""
    _, _ = to_tree(link, rd=True)
    return leaves_list(link).tolist()


def _get_cluster_var(cov: pd.DataFrame, items: list[int]) -> float:
    """Minimum-variance portfolio variance for a sub-cluster."""
    sub_cov = cov.iloc[items, items].values
    n = len(items)
    if n == 1:
        return float(sub_cov[0, 0])
    inv_var = 1.0 / np.maximum(np.diag(sub_cov), 1e-10)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def _recursive_bisect(cov: pd.DataFrame, sorted_items: list[int]) -> pd.Series:
    """Recursive bisection: split into two halves and allocate by inverse cluster variance."""
    weights = pd.Series(1.0, index=sorted_items)
    items_to_bisect = [sorted_items]

    while items_to_bisect:
        # Generate next level of splits
        items_to_bisect = [
            segment[j:k]
            for segment in items_to_bisect
            for j, k in ((0, len(segment) // 2), (len(segment) // 2, len(segment)))
            if len(segment) > 1
        ]
        # Allocate weights between sibling clusters
        for i in range(0, len(items_to_bisect), 2):
            if i + 1 >= len(items_to_bisect):
                break
            left = items_to_bisect[i]
            right = items_to_bisect[i + 1]
            var_left = _get_cluster_var(cov, left)
            var_right = _get_cluster_var(cov, right)
            # Prevent division by zero and ensure allocation stays within [0,1]
            denom = max(var_left + var_right, 1e-10)
            alpha = 1.0 - var_left / denom
            alpha = np.clip(alpha, 0.0, 1.0)
            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)

    return weights


class HRPOptimizer:
    """
    Hierarchical Risk Parity portfolio optimizer.

    The optimizer includes optional filters to improve signal quality:
      * Volatility filter – removes assets whose recent volatility is excessively high.
      * Shrinkage – blends the HRP allocation with an equal‑weight portfolio to temper extreme positions.

    Parameters
    ----------
    max_vol_factor : float, optional
        Maximum allowed volatility as a multiple of the median volatility of the universe.
        Assets with a standard deviation greater than ``median_vol * max_vol_factor`` are excluded.
        Default is ``2.0``.
    shrinkage : float, optional
        Weight applied to an equal‑weight portfolio before normalisation.
        ``0`` means pure HRP, ``1`` means pure equal weighting.
        Default is ``0.1``.
    """

    def __init__(self, max_vol_factor: float = 2.0, shrinkage: float = 0.1) -> None:
        self.max_vol_factor = max_vol_factor
        self.shrinkage = np.clip(shrinkage, 0.0, 1.0)

    def _apply_volatility_filter(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Remove assets whose volatility exceeds the configured threshold."""
        vol = returns.std()
        median_vol = vol.median()
        if median_vol == 0:
            # All assets have zero variance; keep them all
            return returns
        threshold = median_vol * self.max_vol_factor
        mask = vol <= threshold
        filtered = returns.loc[:, mask]
        if filtered.shape[1] < returns.shape[1]:
            dropped = returns.columns[~mask]
            logger.info(
                "HRP volatility filter removed %d assets: %s",
                len(dropped),
                ", ".join(dropped),
            )
        return filtered

    def compute_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute HRP weights for a universe of assets.

        Args
        ----
        returns : pd.DataFrame
            Asset returns with columns = symbols and rows = dates.
            Must contain at least 2 assets and 10 observations.

        Returns
        -------
        pd.Series
            Portfolio weights summing to 1.0, indexed by symbol.
            Falls back to equal weights if data is insufficient or degenerate.
        """
        symbols = list(returns.columns)
        n = len(symbols)

        if n < 2 or len(returns) < 10:
            return pd.Series(1.0 / max(n, 1), index=symbols)

        # Drop columns that are entirely NaN, then fill remaining NaNs with zero.
        returns_clean = returns.dropna(axis=1, how="all").fillna(0.0)
        if returns_clean.shape[1] < 2:
            return pd.Series(1.0 / max(n, 1), index=symbols)

        # Apply volatility filter as a confirmation step.
        returns_filtered = self._apply_volatility_filter(returns_clean)
        symbols_filtered = list(returns_filtered.columns)
        n_filtered = len(symbols_filtered)

        if n_filtered < 2:
            # Not enough assets after filtering – revert to equal weighting.
            logger.warning(
                "HRP optimizer: insufficient assets after volatility filter (n=%d). "
                "Falling back to equal weights.",
                n_filtered,
            )
            return pd.Series(1.0 / max(n, 1), index=symbols)

        try:
            corr = returns_filtered.corr().clip(-0.9999, 0.9999)
            cov = returns_filtered.cov()

            # Convert correlation to distance and perform hierarchical clustering.
            dist = _corr_to_distance(corr)
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method="ward")
            sorted_items = _get_quasi_diag(link)

            # Compute raw HRP weights on the filtered set.
            weights_raw = _recursive_bisect(cov, sorted_items)

            # Map raw weights back to the original symbol list.
            result = pd.Series(0.0, index=symbols)
            for idx, sym in enumerate(symbols_filtered):
                if idx in weights_raw.index:
                    result[sym] = float(weights_raw[idx])

            # Apply shrinkage towards equal weighting.
            if self.shrinkage > 0.0:
                equal_weight = 1.0 / n
                result = (1.0 - self.shrinkage) * result + self.shrinkage * equal_weight

            # Normalise to ensure weights sum to 1.
            total = result.sum()
            if total > 0:
                result = result / total
            else:
                result = pd.Series(1.0 / n, index=symbols)

            return result

        except Exception as exc:  # noqa: BLE001
            # Log the failure and fall back to equal weighting.
            logger.warning("HRP optimization failed — falling back to equal weights (%s)", exc)
            return pd.Series(1.0 / n, index=symbols)