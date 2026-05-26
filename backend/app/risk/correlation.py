"""Detect correlated clusters and enforce allocation limits per cluster."""
import numpy as np
import pandas as pd
from app.utils.logging import logger


def compute_correlation_clusters(returns: pd.DataFrame, threshold: float = 0.70) -> dict[str, list[str]]:
    """
    Group symbols whose 60-day return correlation > threshold into clusters.
    Returns {cluster_id: [symbol1, symbol2, ...]}
    """
    corr = returns.tail(60).corr()
    visited: set[str] = set()
    clusters: dict[str, list[str]] = {}
    cluster_idx = 0

    for sym in corr.columns:
        if sym in visited:
            continue
        cluster = [sym]
        for other in corr.columns:
            if other != sym and other not in visited:
                if abs(corr.loc[sym, other]) > threshold:
                    cluster.append(other)
                    visited.add(other)
        visited.add(sym)
        clusters[f"cluster_{cluster_idx}"] = cluster
        cluster_idx += 1

    return clusters


def check_cluster_limits(
    new_symbol: str,
    new_value_usd: float,
    current_positions: dict[str, float],
    clusters: dict[str, list[str]],
    max_cluster_pct: float = 0.30,
    total_equity: float = 100_000,
) -> tuple[bool, str]:
    """Return (allowed, reason). Blocks if adding new_symbol exceeds max_cluster_pct of equity."""
    for cluster_id, members in clusters.items():
        if new_symbol not in members:
            continue
        cluster_value = sum(current_positions.get(m, 0.0) for m in members) + new_value_usd
        cluster_pct = cluster_value / total_equity
        if cluster_pct > max_cluster_pct:
            reason = f"{new_symbol} would push {cluster_id} to {cluster_pct:.1%} (max {max_cluster_pct:.1%})"
            logger.warning("Cluster limit breached", **{"cluster": cluster_id, "pct": cluster_pct})
            return False, reason
    return True, "ok"
