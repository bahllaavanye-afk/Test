"""Detect correlated clusters and enforce allocation limits per cluster."""
import numbers
import numpy as np
import pandas as pd
from app.utils.logging import logger


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """Union-find connected components for correlation clustering.

    Validation:
        - ``returns`` must be a pandas DataFrame.
        - ``threshold`` must be a float between 0 and 1 (inclusive).

    Raises:
        ValueError: If any input validation fails.
    """
    # Input validation
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("`returns` must be a pandas DataFrame.")
    if not isinstance(threshold, numbers.Real):
        raise ValueError("`threshold` must be a numeric type.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("`threshold` must be between 0 and 1 (inclusive).")

    returns_df = returns.tail(60) if len(returns) > 60 else returns
    symbols = list(returns_df.columns)
    if len(symbols) < 2 or len(returns_df) < 3:
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
                logger.debug(
                    "correlation pair %s/%s skipped: %s", s_a, s_b, exc
                )
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
    """Return (allowed, reason). Blocks if adding new_symbol exceeds max_cluster_pct of equity.

    Validation:
        - ``new_symbol`` must be a non‑empty string.
        - ``new_value_usd`` must be a non‑negative number.
        - ``current_positions`` must be a dict mapping strings to numeric values.
        - ``clusters`` must be a dict mapping strings to lists of strings.
        - ``max_cluster_pct`` must be between 0 and 1 (inclusive).
        - ``total_equity`` must be a positive number.

    Raises:
        ValueError: If any input validation fails.
    """
    # Input validation
    if not isinstance(new_symbol, str) or not new_symbol:
        raise ValueError("`new_symbol` must be a non‑empty string.")
    if not isinstance(new_value_usd, numbers.Real):
        raise ValueError("`new_value_usd` must be a numeric type.")
    if new_value_usd < 0:
        raise ValueError("`new_value_usd` cannot be negative.")
    if not isinstance(current_positions, dict):
        raise ValueError("`current_positions` must be a dictionary.")
    for k, v in current_positions.items():
        if not isinstance(k, str):
            raise ValueError("All keys in `current_positions` must be strings.")
        if not isinstance(v, numbers.Real):
            raise ValueError(
                f"Position value for '{k}' must be a numeric type."
            )
    if not isinstance(clusters, dict):
        raise ValueError("`clusters` must be a dictionary.")
    for cid, members in clusters.items():
        if not isinstance(cid, str):
            raise ValueError("All keys in `clusters` must be strings.")
        if not isinstance(members, list):
            raise ValueError(f"Cluster '{cid}' members must be a list of strings.")
        for m in members:
            if not isinstance(m, str):
                raise ValueError(f"Cluster member '{m}' in '{cid}' must be a string.")
    if not isinstance(max_cluster_pct, numbers.Real):
        raise ValueError("`max_cluster_pct` must be a numeric type.")
    if not (0.0 <= max_cluster_pct <= 1.0):
        raise ValueError("`max_cluster_pct` must be between 0 and 1 (inclusive).")
    if not isinstance(total_equity, numbers.Real):
        raise ValueError("`total_equity` must be a numeric type.")
    if total_equity <= 0:
        raise ValueError("`total_equity` must be greater than zero.")

    for cluster_id, members in clusters.items():
        if new_symbol not in members:
            continue
        cluster_value = sum(
            current_positions.get(m, 0.0) for m in members
        ) + new_value_usd
        cluster_pct = cluster_value / total_equity
        if cluster_pct > max_cluster_pct:
            reason = (
                f"{new_symbol} would push {cluster_id} to "
                f"{cluster_pct:.1%} (max {max_cluster_pct:.1%})"
            )
            logger.warning(
                "Cluster limit breached", **{"cluster": cluster_id, "pct": cluster_pct}
            )
            return False, reason
    return True, "ok"