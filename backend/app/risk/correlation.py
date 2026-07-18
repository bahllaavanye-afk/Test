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
            "compute_correlation_clusters skipped",
            **{
                "signal_count": len(symbols),
                "elapsed_ms": (time.perf_counter() - start_time) * 1000,
                "clusters": 0,
            },
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
        for s_b in symbols[i + 1 :]:
            try:
                corr = corr_matrix.loc[s_a, s_b]
                if abs(corr) > threshold:
                    union(s_a, s_b)
            except Exception as exc:  # noqa: BLE001 — pair missing from matrix
                logger.debug("correlation pair %s/%s skipped: %s", s_a, s_b, exc)
                continue

    clusters: dict[str, list[str]] = {}
    for s in symbols:
        root = find(s)
        clusters.setdefault(root, []).append(s)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "compute_correlation_clusters completed",
        **{
            "signal_count": len(symbols),
            "cluster_count": len(clusters),
            "elapsed_ms": elapsed_ms,
        },
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
    start_time = time.perf_counter()
    for cluster_id, members in clusters.items():
        if new_symbol not in members:
            continue
        cluster_value = sum(current_positions.get(m, 0.0) for m in members) + new_value_usd
        cluster_pct = cluster_value / total_equity
        if cluster_pct > max_cluster_pct:
            reason = f"{new_symbol} would push {cluster_id} to {cluster_pct:.1%} (max {max_cluster_pct:.1%})"
            logger.warning(
                "Cluster limit breached",
                **{"cluster": cluster_id, "pct": cluster_pct, "new_value_usd": new_value_usd},
            )
            logger.info(
                "check_cluster_limits completed",
                **{
                    "signal": new_symbol,
                    "elapsed_ms": (time.perf_counter() - start_time) * 1000,
                    "allowed": False,
                },
            )
            return False, reason
    logger.info(
        "check_cluster_limits completed",
        **{
            "signal": new_symbol,
            "elapsed_ms": (time.perf_counter() - start_time) * 1000,
            "allowed": True,
        },
    )
    return True, "ok"