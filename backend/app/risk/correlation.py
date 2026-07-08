"""Detect correlated clusters and enforce allocation limits per cluster."""
import time
import numpy as np
import pandas as pd
from app.utils.logging import logger


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """Union-find connected components for correlation clustering."""
    start_time = time.perf_counter()
    returns_df = returns.tail(60) if len(returns) > 60 else returns
    symbols = list(returns_df.columns)
    if len(symbols) < 2 or len(returns_df) < 3:
        logger.info(
            "Correlation clustering skipped",
            signal_count=len(symbols),
            execution_time=time.perf_counter() - start_time,
        )
        return {}

    parent = {s: s for s in symbols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    corr_matrix = returns_df.corr()
    for i, s_a in enumerate(symbols):
        for s_b in symbols[i + 1:]:
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

    duration = time.perf_counter() - start_time
    logger.info(
        "Correlation clusters computed",
        signal_count=len(symbols),
        cluster_count=len(clusters),
        execution_time=duration,
    )
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
            logger.warning(
                "Cluster limit breached",
                cluster=cluster_id,
                pct=cluster_pct,
                new_symbol=new_symbol,
                new_value_usd=new_value_usd,
            )
            logger.info(
                "Cluster limit check result",
                symbol=new_symbol,
                allowed=False,
                cluster=cluster_id,
                cluster_pct=cluster_pct,
                execution_time=0.0,
            )
            return False, reason
    logger.info(
        "Cluster limit check result",
        symbol=new_symbol,
        allowed=True,
        cluster=None,
        cluster_pct=0.0,
        execution_time=0.0,
    )
    return True, "ok"