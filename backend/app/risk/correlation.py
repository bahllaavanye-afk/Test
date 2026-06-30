"""Detect correlated clusters and enforce allocation limits per cluster."""
import numpy as np
import pandas as pd
from app.utils.logging import logger
from typing import Dict, List, Tuple


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> Dict[str, List[str]]:
    """Union‑find connected components for correlation clustering.

    Args:
        returns: DataFrame where columns represent symbols and rows represent returns.
        threshold: Correlation magnitude above which symbols are considered linked.

    Returns:
        A mapping from cluster root symbol to the list of symbols in that cluster.

    Raises:
        TypeError: If ``returns`` is not a ``pd.DataFrame``.
        ValueError: If ``threshold`` is not in the interval (0, 1].
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(returns, pd.DataFrame):
        logger.error("Invalid input type for returns", expected_type="pd.DataFrame", actual_type=type(returns))
        raise TypeError("returns must be a pandas DataFrame")

    if not isinstance(threshold, (float, int)):
        logger.error("Invalid input type for threshold", expected_type="float", actual_type=type(threshold))
        raise TypeError("threshold must be a float")

    if not (0 < float(threshold) <= 1):
        logger.error("Threshold out of bounds", threshold=threshold)
        raise ValueError("threshold must be in the interval (0, 1]")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    returns_df = returns.tail(60) if len(returns) > 60 else returns
    symbols = list(returns_df.columns)

    # Not enough data to form clusters
    if len(symbols) < 2 or len(returns_df) < 3:
        return {}

    parent: Dict[str, str] = {s: s for s in symbols}

    def find(x: str) -> str:
        """Find the root of the element ``x`` with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        """Union the sets containing ``x`` and ``y``."""
        parent[find(x)] = find(y)

    # Compute correlation matrix; any failure is logged and re‑raised.
    try:
        corr_matrix = returns_df.corr()
    except Exception as exc:
        logger.exception("Failed to compute correlation matrix", exc=exc)
        raise

    for i, s_a in enumerate(symbols):
        for s_b in symbols[i + 1 :]:
            try:
                corr = corr_matrix.loc[s_a, s_b]
                # NaN correlations are ignored
                if pd.isna(corr):
                    continue
                if abs(corr) > threshold:
                    union(s_a, s_b)
            except KeyError as ke:
                logger.error(
                    "Correlation lookup failed",
                    symbol_a=s_a,
                    symbol_b=s_b,
                    error=str(ke),
                )
                continue
            except Exception as exc:
                logger.exception(
                    "Unexpected error during correlation clustering",
                    symbol_a=s_a,
                    symbol_b=s_b,
                    error=exc,
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
    """Return ``(allowed, reason)``. Blocks if adding ``new_symbol`` exceeds
    ``max_cluster_pct`` of equity.

    Args:
        new_symbol: Symbol being added.
        new_value_usd: Dollar value of the new position.
        current_positions: Mapping from symbol to its current dollar exposure.
        clusters: Mapping from cluster identifier to member symbols.
        max_cluster_pct: Maximum allowed proportion of equity per cluster.
        total_equity: Total portfolio equity.

    Returns:
        Tuple where the first element indicates if the addition is allowed,
        and the second element provides a human‑readable reason.

    Raises:
        TypeError: For incorrect argument types.
        ValueError: If ``max_cluster_pct`` is not in (0, 1] or ``total_equity`` <= 0.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(new_symbol, str):
        logger.error("Invalid type for new_symbol", expected_type="str", actual_type=type(new_symbol))
        raise TypeError("new_symbol must be a string")

    if not isinstance(new_value_usd, (float, int)):
        logger.error("Invalid type for new_value_usd", expected_type="float", actual_type=type(new_value_usd))
        raise TypeError("new_value_usd must be a numeric type")

    if not isinstance(current_positions, dict):
        logger.error("Invalid type for current_positions", expected_type="dict", actual_type=type(current_positions))
        raise TypeError("current_positions must be a dict")

    if not isinstance(clusters, dict):
        logger.error("Invalid type for clusters", expected_type="dict", actual_type=type(clusters))
        raise TypeError("clusters must be a dict")

    if not isinstance(max_cluster_pct, (float, int)):
        logger.error("Invalid type for max_cluster_pct", expected_type="float", actual_type=type(max_cluster_pct))
        raise TypeError("max_cluster_pct must be a numeric type")

    if not (0 < float(max_cluster_pct) <= 1):
        logger.error("max_cluster_pct out of bounds", max_cluster_pct=max_cluster_pct)
        raise ValueError("max_cluster_pct must be in the interval (0, 1]")

    if not isinstance(total_equity, (float, int)):
        logger.error("Invalid type for total_equity", expected_type="float", actual_type=type(total_equity))
        raise TypeError("total_equity must be a numeric type")

    if total_equity <= 0:
        logger.error("Non‑positive total_equity", total_equity=total_equity)
        raise ValueError("total_equity must be greater than zero")

    # ------------------------------------------------------------------
    # Core logic with safe arithmetic
    # ------------------------------------------------------------------
    for cluster_id, members in clusters.items():
        if new_symbol not in members:
            continue
        try:
            cluster_value = sum(current_positions.get(m, 0.0) for m in members) + new_value_usd
            cluster_pct = cluster_value / total_equity
        except Exception as exc:
            logger.exception(
                "Error computing cluster limits",
                cluster_id=cluster_id,
                members=members,
                new_symbol=new_symbol,
                new_value_usd=new_value_usd,
                error=exc,
            )
            raise

        if cluster_pct > max_cluster_pct:
            reason = (
                f"{new_symbol} would push {cluster_id} to {cluster_pct:.1%} "
                f"(max {max_cluster_pct:.1%})"
            )
            logger.warning(
                "Cluster limit breached",
                cluster=cluster_id,
                pct=cluster_pct,
                limit=max_cluster_pct,
                symbol=new_symbol,
            )
            return False, reason

    return True, "ok"