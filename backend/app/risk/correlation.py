"""Detect correlated clusters and enforce allocation limits per cluster.

This module provides utilities to identify groups of instruments whose returns
are strongly correlated and to ensure that the aggregate exposure of any such
cluster does not exceed a configurable percentage of the total equity.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from app.utils.logging import logger


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> Dict[str, List[str]]:
    """Identify correlation clusters using a union‑find algorithm.

    The function examines the most recent 60 rows of ``returns`` (or the full
    DataFrame if it contains fewer than 60 rows) and builds an undirected graph
    where an edge exists between two symbols when the absolute Pearson
    correlation exceeds ``threshold``. Connected components of this graph are
    returned as clusters.

    Parameters
    ----------
    returns : pd.DataFrame
        A DataFrame where each column represents a symbol and each row a
        sequential return observation.
    threshold : float, optional
        Correlation magnitude above which two symbols are considered linked.
        Defaults to ``0.70``.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from a cluster identifier (the root symbol) to the list of
        symbols belonging to that cluster. An empty dictionary is returned when
        there are fewer than two symbols or insufficient observations.
    """
    returns_df = returns.tail(60) if len(returns) > 60 else returns
    symbols = list(returns_df.columns)
    if len(symbols) < 2 or len(returns_df) < 3:
        return {}

    parent: Dict[str, str] = {s: s for s in symbols}

    def find(x: str) -> str:
        """Find the root of ``x`` with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        """Merge the sets containing ``x`` and ``y``."""
        parent[find(x)] = find(y)

    corr_matrix = returns_df.corr()
    for i, s_a in enumerate(symbols):
        for s_b in symbols[i + 1 :]:
            try:
                corr = corr_matrix.loc[s_a, s_b]
                if abs(corr) > threshold:
                    union(s_a, s_b)
            except Exception as exc:  # noqa: BLE001 – pair missing from matrix
                logger.debug(
                    "correlation pair %s/%s skipped: %s", s_a, s_b, exc
                )
                continue

    clusters: Dict[str, List[str]] = {}
    for s in symbols:
        root = find(s)
        clusters.setdefault(root, []).append(s)
    return clusters


def check_cluster_limits(
    new_symbol: str,
    new_value_usd: float,
    current_positions: Dict[str, float],
    clusters: Dict[str, List[str]],
    max_cluster_pct: float = 0.30,
    total_equity: float = 100_000,
) -> Tuple[bool, str]:
    """Determine whether adding a position respects cluster exposure limits.

    The function evaluates the cluster that contains ``new_symbol``. It adds the
    prospective ``new_value_usd`` to the existing exposure of that cluster and
    checks whether the resulting percentage of ``total_equity`` exceeds
    ``max_cluster_pct``. If the limit would be breached, ``allowed`` is ``False``
    and a descriptive reason is returned; otherwise ``allowed`` is ``True`` with
    a generic ``"ok"`` reason.

    Parameters
    ----------
    new_symbol : str
        Symbol of the prospective position.
    new_value_usd : float
        Dollar value of the prospective position.
    current_positions : Dict[str, float]
        Mapping of existing symbols to their dollar exposure.
    clusters : Dict[str, List[str]]
        Correlation clusters as produced by :func:`compute_correlation_clusters`.
    max_cluster_pct : float, optional
        Maximum allowed exposure per cluster as a fraction of total equity.
        Defaults to ``0.30`` (30 %).
    total_equity : float, optional
        The portfolio's total equity against which percentages are computed.
        Defaults to ``100_000``.

    Returns
    -------
    Tuple[bool, str]
        ``(allowed, reason)`` where ``allowed`` indicates if the position can be
        added and ``reason`` explains the decision.
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
                "Cluster limit breached", **{"cluster": cluster_id, "pct": cluster_pct}
            )
            return False, reason
    return True, "ok"