"""Detect correlated clusters and enforce allocation limits per cluster with enhanced entry and exit logic."""
import numpy as np
import pandas as pd
from app.utils.logging import logger
from typing import Dict, List, Tuple


def _rolling_correlation(
    returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Calculate rolling correlation matrix over the given window."""
    # Use pandas rolling with pairwise correlation; fallback to simple corr if insufficient data.
    if len(returns) < window:
        return returns.corr()
    rolled = returns.rolling(window).corr()
    # The result is a MultiIndex (date, symbol1) with columns symbol2.
    # We take the last available correlation matrix.
    last_date = rolled.index[-1]
    corr_matrix = rolled.loc[last_date]
    # Fill NaNs with 0 (no correlation) to avoid spurious unions.
    return corr_matrix.fillna(0.0)


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
    confirmation_window: int = 60,
    confirmation_required: int = 2,
) -> Dict[str, List[str]]:
    """
    Identify clusters of highly correlated symbols using a union‑find approach.

    Entry is tightened by requiring the correlation to exceed ``threshold`` for
    ``confirmation_required`` consecutive windows of ``confirmation_window`` days.
    This reduces false positives caused by transient spikes.

    Parameters
    ----------
    returns: pd.DataFrame
        Historical returns indexed by date with symbols as columns.
    threshold: float, default 0.70
        Minimum absolute correlation to consider symbols linked.
    confirmation_window: int, default 60
        Length of each rolling window (in days) used for correlation calculation.
    confirmation_required: int, default 2
        Number of consecutive windows that must meet the threshold.

    Returns
    -------
    dict[str, list[str]]
        Mapping from cluster root symbol to list of member symbols.
    """
    if returns.empty:
        logger.debug("Empty returns DataFrame supplied to compute_correlation_clusters.")
        return {}

    # Prepare rolling correlation matrices.
    rolling_corrs: List[pd.DataFrame] = []
    for i in range(confirmation_required):
        start_idx = max(0, len(returns) - (i + 1) * confirmation_window)
        window_df = returns.iloc[start_idx : start_idx + confirmation_window]
        if len(window_df) < 2:
            logger.debug(
                "Insufficient data for rolling window %d (size %d). Skipping.",
                i,
                len(window_df),
            )
            continue
        rolling_corrs.append(_rolling_correlation(window_df, confirmation_window))

    if not rolling_corrs:
        logger.debug("No valid rolling correlation windows could be computed.")
        return {}

    # Intersection of correlation masks across required windows.
    mask = np.ones_like(rolling_corrs[0].values, dtype=bool)
    for corr_mat in rolling_corrs:
        mask &= (corr_mat.abs().values > threshold)

    # Build union‑find structure based on the intersected mask.
    symbols = list(returns.columns)
    if len(symbols) < 2:
        return {}

    parent: Dict[str, str] = {s: s for s in symbols}

    def find(x: str) -> str:
        """Find root of the disjoint set containing x with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        """Union the sets containing x and y."""
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    # Apply unions where mask is True.
    for i, s_a in enumerate(symbols):
        for j, s_b in enumerate(symbols[i + 1 :], start=i + 1):
            if mask[i, j]:
                union(s_a, s_b)

    # Assemble clusters.
    clusters: Dict[str, List[str]] = {}
    for s in symbols:
        root = find(s)
        clusters.setdefault(root, []).append(s)

    logger.debug("Computed %d correlation clusters.", len(clusters))
    return clusters


def check_cluster_limits(
    new_symbol: str,
    new_value_usd: float,
    current_positions: Dict[str, float],
    clusters: Dict[str, List[str]],
    max_cluster_pct: float = 0.30,
    total_equity: float = 100_000,
) -> Tuple[bool, str]:
    """
    Validate that adding a new position does not breach the cluster exposure limit.

    The function now also provides a suggested reduction amount when the limit would be exceeded,
    enabling a more graceful exit strategy.

    Parameters
    ----------
    new_symbol: str
        Symbol of the prospective position.
    new_value_usd: float
        Dollar value of the prospective position.
    current_positions: dict[str, float]
        Existing position values keyed by symbol.
    clusters: dict[str, list[str]]
        Correlation clusters as returned by ``compute_correlation_clusters``.
    max_cluster_pct: float, default 0.30
        Maximum allowed proportion of equity allocated to any single cluster.
    total_equity: float, default 100_000
        Total equity against which percentages are measured.

    Returns
    -------
    tuple[bool, str]
        ``(allowed, reason)`` where ``allowed`` indicates if the addition is permissible.
        ``reason`` contains an explanatory message; if not allowed, it also suggests a reduction.
    """
    for cluster_id, members in clusters.items():
        if new_symbol not in members:
            continue

        cluster_value = sum(current_positions.get(m, 0.0) for m in members) + new_value_usd
        cluster_pct = cluster_value / total_equity

        if cluster_pct > max_cluster_pct:
            excess_pct = cluster_pct - max_cluster_pct
            excess_usd = excess_pct * total_equity
            # Suggest reducing the new position first, then existing ones if needed.
            suggested_reduction = min(new_value_usd, excess_usd)
            reason = (
                f"{new_symbol} would push cluster {cluster_id} to {cluster_pct:.1%} "
                f"(max {max_cluster_pct:.1%}). Reduce new allocation by ${suggested_reduction:,.2f}."
            )
            logger.warning(
                "Cluster limit breached",
                extra={"cluster": cluster_id, "pct": cluster_pct, "suggested_reduction": suggested_reduction},
            )
            return False, reason

    return True, "ok"


def evaluate_cluster_exit(
    symbol: str,
    current_positions: Dict[str, float],
    clusters: Dict[str, List[str]],
    max_cluster_pct: float = 0.30,
    total_equity: float = 100_000,
) -> Tuple[bool, str]:
    """
    Determine if an existing position should be exited (or reduced) due to cluster over‑allocation.

    This acts as a confirmation filter for exits: if a cluster exceeds the threshold,
    the function recommends scaling down the offending symbols.

    Parameters
    ----------
    symbol: str
        Symbol under evaluation.
    current_positions: dict[str, float]
        Current portfolio allocations.
    clusters: dict[str, list[str]]
        Correlation clusters.
    max_cluster_pct: float, default 0.30
        Maximum allowed cluster exposure.
    total_equity: float, default 100_000
        Portfolio equity.

    Returns
    -------
    tuple[bool, str]
        ``(should_exit, message)`` where ``should_exit`` is True if the position breaches
        cluster limits, and ``message`` provides guidance.
    """
    for cluster_id, members in clusters.items():
        if symbol not in members:
            continue

        cluster_value = sum(current_positions.get(m, 0.0) for m in members)
        cluster_pct = cluster_value / total_equity

        if cluster_pct > max_cluster_pct:
            excess_pct = cluster_pct - max_cluster_pct
            excess_usd = excess_pct * total_equity
            # Proportionally allocate reduction across members.
            total_member_value = sum(current_positions.get(m, 0.0) for m in members)
            if total_member_value == 0:
                reduction = 0.0
            else:
                reduction = (current_positions.get(symbol, 0.0) / total_member_value) * excess_usd

            message = (
                f"Cluster {cluster_id} exceeds limit at {cluster_pct:.1%}. "
                f"Recommend reducing {symbol} by ${reduction:,.2f}."
            )
            logger.info(
                "Cluster exit recommendation",
                extra={"cluster": cluster_id, "symbol": symbol, "reduction": reduction},
            )
            return True, message

    return False, "cluster within limits"