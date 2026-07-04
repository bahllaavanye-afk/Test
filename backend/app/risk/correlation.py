"""Detect correlated clusters and enforce allocation limits per cluster."""
import numpy as np
import pandas as pd
from app.utils.logging import logger


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """Union-find connected components for correlation clustering.

    Args:
        returns: DataFrame containing asset return series. Must have at least one column.
        threshold: Correlation magnitude threshold for linking assets. Must be >0 and <=1.

    Returns:
        A dictionary mapping cluster root identifiers to lists of member symbols.

    Raises:
        ValueError: If `returns` is not a pandas DataFrame, is empty, or if `threshold`
            is not a float/int in the (0, 1] range.
    """
    # Input validation
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("`returns` must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("`returns` DataFrame must contain at least one row and one column.")
    if not isinstance(threshold, (float, int)):
        raise ValueError("`threshold` must be a numeric type (float or int).")
    if not (0 < threshold <= 1):
        raise ValueError("`threshold` must be greater than 0 and at most 1.")

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
    """Return (allowed, reason). Blocks if adding new_symbol exceeds max_cluster_pct of equity.

    Args:
        new_symbol: Symbol to be added. Must be a non‑empty string.
        new_value_usd: Position size in USD for the new symbol. Must be non‑negative.
        current_positions: Mapping of existing symbols to their USD values.
        clusters: Mapping of cluster identifiers to lists of member symbols.
        max_cluster_pct: Maximum allowed proportion of equity per cluster (0 < pct <= 1).
        total_equity: Total portfolio equity in USD. Must be positive.

    Returns:
        Tuple where the first element is a boolean indicating if the addition is allowed,
        and the second element is a descriptive reason.

    Raises:
        ValueError: If any input is of incorrect type or out of expected bounds.
    """
    # Input validation
    if not isinstance(new_symbol, str) or not new_symbol:
        raise ValueError("`new_symbol` must be a non‑empty string.")
    if not isinstance(new_value_usd, (float, int)):
        raise ValueError("`new_value_usd` must be a numeric type (float or int).")
    if new_value_usd < 0:
        raise ValueError("`new_value_usd` cannot be negative.")
    if not isinstance(current_positions, dict):
        raise ValueError("`current_positions` must be a dictionary mapping symbols to float values.")
    for k, v in current_positions.items():
        if not isinstance(k, str):
            raise ValueError("Keys in `current_positions` must be strings.")
        if not isinstance(v, (float, int)):
            raise ValueError(f"Value for symbol '{k}' in `current_positions` must be numeric.")
    if not isinstance(clusters, dict):
        raise ValueError("`clusters` must be a dictionary mapping cluster IDs to lists of symbols.")
    for cid, members in clusters.items():
        if not isinstance(cid, str):
            raise ValueError("Cluster IDs in `clusters` must be strings.")
        if not isinstance(members, list):
            raise ValueError(f"Members of cluster '{cid}' must be provided as a list.")
        for m in members:
            if not isinstance(m, str):
                raise ValueError(f"Each member in cluster '{cid}' must be a string.")
    if not isinstance(max_cluster_pct, (float, int)):
        raise ValueError("`max_cluster_pct` must be a numeric type (float or int).")
    if not (0 < max_cluster_pct <= 1):
        raise ValueError("`max_cluster_pct` must be greater than 0 and at most 1.")
    if not isinstance(total_equity, (float, int)):
        raise ValueError("`total_equity` must be a numeric type (float or int).")
    if total_equity <= 0:
        raise ValueError("`total_equity` must be a positive value.")

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
            logger.warning("Cluster limit breached", **{"cluster": cluster_id, "pct": cluster_pct})
            return False, reason
    return True, "ok"