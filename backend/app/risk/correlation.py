"""Detect correlated clusters and enforce allocation limits per cluster."""
import numpy as np
import pandas as pd
from app.utils.logging import logger


def _prepare_returns_df(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Trim returns to the most recent ``window`` rows if longer."""
    return returns.tail(window) if len(returns) > window else returns


def _find(parent: dict[str, str], x: str) -> str:
    """Find the root of ``x`` with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: dict[str, str], x: str, y: str) -> None:
    """Union the sets containing ``x`` and ``y``."""
    parent[_find(parent, x)] = _find(parent, y)


def _populate_union_find(
    parent: dict[str, str],
    symbols: list[str],
    corr_matrix: pd.DataFrame,
    threshold: float,
) -> None:
    """Link symbols whose correlation magnitude exceeds ``threshold``."""
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


def _extract_clusters(
    parent: dict[str, str], symbols: list[str]
) -> dict[str, list[str]]:
    """Collect symbols into clusters based on their union‑find roots."""
    clusters: dict[str, list[str]] = {}
    for s in symbols:
        root = _find(parent, s)
        clusters.setdefault(root, []).append(s)
    return clusters


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """Union‑find connected components for correlation clustering.

    Returns a mapping from cluster identifier to the list of symbols belonging
    to that cluster. Empty dict is returned when insufficient data is available.
    """
    returns_df = _prepare_returns_df(returns)
    symbols = list(returns_df.columns)

    if len(symbols) < 2 or len(returns_df) < 3:
        return {}

    parent = {s: s for s in symbols}
    corr_matrix = returns_df.corr()

    _populate_union_find(parent, symbols, corr_matrix, threshold)
    return _extract_clusters(parent, symbols)


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
            reason = (
                f"{new_symbol} would push {cluster_id} to {cluster_pct:.1%} "
                f"(max {max_cluster_pct:.1%})"
            )
            logger.warning(
                "Cluster limit breached", **{"cluster": cluster_id, "pct": cluster_pct}
            )
            return False, reason
    return True, "ok"