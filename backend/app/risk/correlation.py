"""Detect correlated clusters and enforce allocation limits per cluster."""
import numpy as np
import pandas as pd
from app.utils.logging import logger
from typing import Mapping


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """Union‑find connected components for correlation clustering.

    Args:
        returns: DataFrame where columns are symbols and rows are returns.
        threshold: Correlation magnitude above which two symbols are considered linked.

    Returns:
        A mapping from cluster representative to list of member symbols.

    Raises:
        ValueError: If ``returns`` is not a DataFrame or ``threshold`` is outside [0, 1].
    """
    if not isinstance(returns, pd.DataFrame):
        logger.error(
            "Invalid type for returns",
            expected="pd.DataFrame",
            actual=type(returns).__name__,
        )
        raise ValueError("returns must be a pandas DataFrame")
    if not (0.0 <= threshold <= 1.0):
        logger.error(
            "Threshold out of bounds",
            threshold=threshold,
            valid_range="0.0‑1.0",
        )
        raise ValueError("threshold must be between 0 and 1")

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
                if pd.isna(corr):
                    continue
                if abs(corr) > threshold:
                    union(s_a, s_b)
            except KeyError as e:
                logger.error(
                    "Correlation lookup failed",
                    missing_symbol=str(e),
                    symbol_a=s_a,
                    symbol_b=s_b,
                )
                continue
            except Exception as e:
                logger.exception(
                    "Unexpected error during correlation clustering",
                    symbol_a=s_a,
                    symbol_b=s_b,
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
    current_positions: Mapping[str, float],
    clusters: Mapping[str, list[str]],
    max_cluster_pct: float = 0.30,
    total_equity: float = 100_000,
) -> tuple[bool, str]:
    """Return (allowed, reason). Blocks if adding ``new_symbol`` exceeds ``max_cluster_pct`` of equity.

    Args:
        new_symbol: Symbol to be added.
        new_value_usd: Position size in USD for ``new_symbol``.
        current_positions: Mapping of existing symbol → position size in USD.
        clusters: Mapping of cluster identifier → list of member symbols.
        max_cluster_pct: Maximum allowed proportion of total equity for any cluster.
        total_equity: Portfolio equity used for percentage calculations.

    Returns:
        Tuple where the first element indicates allowance and the second provides a reason.

    Raises:
        ValueError: If inputs are of incorrect type or out of expected ranges.
    """
    if not isinstance(new_symbol, str):
        logger.error("new_symbol must be a string", provided_type=type(new_symbol).__name__)
        raise ValueError("new_symbol must be a string")
    if not isinstance(new_value_usd, (int, float)):
        logger.error(
            "new_value_usd must be numeric",
            provided_type=type(new_value_usd).__name__,
        )
        raise ValueError("new_value_usd must be a numeric type")
    if new_value_usd < 0:
        logger.error("new_value_usd cannot be negative", value=new_value_usd)
        raise ValueError("new_value_usd cannot be negative")
    if not isinstance(current_positions, Mapping):
        logger.error(
            "current_positions must be a mapping",
            provided_type=type(current_positions).__name__,
        )
        raise ValueError("current_positions must be a mapping")
    if not isinstance(clusters, Mapping):
        logger.error(
            "clusters must be a mapping",
            provided_type=type(clusters).__name__,
        )
        raise ValueError("clusters must be a mapping")
    if not (0.0 < max_cluster_pct <= 1.0):
        logger.error(
            "max_cluster_pct out of bounds",
            max_cluster_pct=max_cluster_pct,
            valid_range="(0.0, 1.0]",
        )
        raise ValueError("max_cluster_pct must be in (0, 1]")
    if total_equity <= 0:
        logger.error("total_equity must be positive", total_equity=total_equity)
        raise ValueError("total_equity must be positive")

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
                cluster=cluster_id,
                pct=cluster_pct,
                symbol=new_symbol,
                attempted_value=new_value_usd,
            )
            return False, reason
    return True, "ok"