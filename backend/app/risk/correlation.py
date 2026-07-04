"""Detect correlated clusters and enforce allocation limits per cluster.

This module provides utilities to identify groups of assets whose returns are
highly correlated and to enforce portfolio allocation limits on those groups.
The clustering is performed using a simple union‑find algorithm on the
correlation matrix of recent returns.  The limit‑checking function can be used
by the risk engine to ensure that no single correlation cluster exceeds a
configured percentage of the total equity.
"""

import numpy as np
import pandas as pd
from app.utils.logging import logger


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """Identify clusters of highly correlated assets.

    The function looks at the most recent 60 rows of ``returns`` (or all rows
    if fewer than 60 are available) and computes the pairwise correlation
    matrix.  Assets whose absolute correlation exceeds ``threshold`` are
    merged into the same cluster using a union‑find data structure.

    Parameters
    ----------
    returns : pd.DataFrame
        A DataFrame where each column corresponds to an asset and each row
        contains its return for a given period.
    threshold : float, optional
        Correlation magnitude above which two assets are considered linked.
        Default is ``0.70``.

    Returns
    -------
    dict[str, list[str]]
        Mapping from a cluster identifier (the root symbol) to a list of
        symbols belonging to that cluster.  Empty dict if there are fewer than
        two symbols or insufficient data.
    """
    returns_df = returns.tail(60) if len(returns) > 60 else returns
    symbols = list(returns_df.columns)
    if len(symbols) < 2 or len(returns_df) < 3:
        return {}

    parent: dict[str, str] = {s: s for s in symbols}

    def find(x: str) -> str:
        """Find the root of ``x`` with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        """Union the sets containing ``x`` and ``y``."""
        parent[find(x)] = find(y)

    corr_matrix = returns_df.corr()
    for i, s_a in enumerate(symbols):
        for s_b in symbols[i + 1 :]:
            try:
                corr = corr_matrix.loc[s_a, s_b]
                if abs(corr) > threshold:
                    union(s_a, s_b)
            except Exception:
                continue

    clusters: dict[str, list[str]] = {}
    for s in symbols:
        root = find(s)
        clusters.setdefault(root, []).append(s)
    return clusters


def check_cluster_limits(
    new_symbol: str,
    new_value_usd: float,
    current_positions: dict[str, float],
    clusters: dict[str, list[str]],
    max_cluster_pct: float = 0.30,
    total_equity: float = 100_000,
) -> tuple[bool, str]:
    """Determine whether adding a position respects cluster allocation limits.

    The function calculates the total exposure of the cluster that ``new_symbol``
    belongs to, including the proposed ``new_value_usd``.  If the resulting
    exposure exceeds ``max_cluster_pct`` of ``total_equity``, the addition is
    rejected.

    Parameters
    ----------
    new_symbol : str
        Symbol of the asset being added.
    new_value_usd : float
        Dollar value of the proposed addition.
    current_positions : dict[str, float]
        Mapping from symbol to current dollar position.
    clusters : dict[str, list[str]]
        Output of :func:`compute_correlation_clusters`.
    max_cluster_pct : float, optional
        Maximum allowed proportion of equity for any single cluster.
        Default is ``0.30`` (30%).
    total_equity : float, optional
        Total portfolio equity against which percentages are measured.
        Default is ``100_000``.

    Returns
    -------
    tuple[bool, str]
        ``(allowed, reason)`` where ``allowed`` is ``True`` if the addition
        respects the limit, otherwise ``False``.  ``reason`` provides a short
        explanation; it is ``"ok"`` when ``allowed`` is ``True``.
    """
    for cluster_id, members in clusters.items():
        if new_symbol not in members:
            continue
        cluster_value = sum(current_positions.get(m, 0.0) for m in members) + new_value_usd
        cluster_pct = cluster_value / total_equity
        if cluster_pct > max_cluster_pct:
            reason = (
                f"{new_symbol} would push {cluster_id} to {cluster_pct:.1%} "
                f"(max {max_cluster_pct:.1%})"
            )
            logger.warning(
                "Cluster limit breached",
                **{"cluster": cluster_id, "pct": cluster_pct},
            )
            return False, reason
    return True, "ok"