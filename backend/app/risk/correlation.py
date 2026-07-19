"""Detect correlated clusters and enforce allocation limits per cluster."""
import numpy as np
import pandas as pd
from app.utils.logging import logger
from typing import Callable, Dict, List


def _initialize_parent(symbols: List[str]) -> Dict[str, str]:
    """Create a disjoint‑set parent mapping where each symbol is its own root."""
    return {s: s for s in symbols}


def _find(parent: Dict[str, str], x: str) -> str:
    """Find the root of *x* with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: Dict[str, str], x: str, y: str) -> None:
    """Union the sets containing *x* and *y*."""
    parent[_find(parent, x)] = _find(parent, y)


def _build_clusters(symbols: List[str], find_fn: Callable[[str], str]) -> Dict[str, List[str]]:
    """Group symbols by their root identifiers returned by *find_fn*."""
    clusters: Dict[str, List[str]] = {}
    for s in symbols:
        root = find_fn(s)
        clusters.setdefault(root, []).append(s)
    return clusters


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """Union‑find connected components for correlation clustering.

    The function looks at the most recent 60 rows (or fewer if unavailable) of
    ``returns`` and groups assets whose absolute correlation exceeds *threshold*.
    """
    # Use only the latest 60 observations to keep the correlation matrix manageable.
    recent_returns = returns.tail(60) if len(returns) > 60 else returns
    symbols = list(recent_returns.columns)

    # Early exit when clustering is not meaningful.
    if len(symbols) < 2 or len(recent_returns) < 3:
        return {}

    parent = _initialize_parent(symbols)

    corr_matrix = recent_returns.corr()
    for i, s_a in enumerate(symbols):
        for s_b in symbols[i + 1 :]:
            try:
                corr = corr_matrix.loc[s_a, s_b]
                if abs(corr) > threshold:
                    _union(parent, s_a, s_b)
            except Exception as exc:  # noqa: BLE001 — pair missing from matrix
                logger.debug(
                    "correlation pair %s/%s skipped: %s", s_a, s_b, exc
                )
                continue

    # Helper that binds the current ``parent`` mapping to the ``_find`` implementation.
    def find(symbol: str) -> str:
        return _find(parent, symbol)

    return _build_clusters(symbols, find)


def check_cluster_limits(
    new_symbol: str,
    new_value_usd: float,
    current_positions: dict[str, float],
    clusters: dict[str, list[str]],
    max_cluster_pct: float = 0.30,
    total_equity: float = 100_000,
) -> tuple[bool, str]:
    """Return (allowed, reason). Blocks if adding *new_symbol* exceeds *max_cluster_pct* of equity."""
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