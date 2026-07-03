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

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns where columns are symbols.
    threshold : float, optional
        Correlation magnitude above which assets are considered linked,
        by default 0.70.

    Returns
    -------
    dict[str, list[str]]
        Mapping from cluster root symbol to list of member symbols.
    """
    # Input validation
    if not isinstance(returns, pd.DataFrame):
        logger.error(
            "Invalid returns type",
            expected_type="pd.DataFrame",
            actual_type=type(returns).__name__,
        )
        raise TypeError("returns must be a pandas DataFrame")
    if not isinstance(threshold, (float, int)):
        logger.error(
            "Invalid threshold type",
            expected_type="float or int",
            actual_type=type(threshold).__name__,
        )
        raise TypeError("threshold must be a numeric type")
    if not (0.0 <= float(threshold) <= 1.0):
        logger.error(
            "Threshold out of bounds",
            threshold=threshold,
        )
        raise ValueError("threshold must be between 0 and 1")

    # Use the most recent 60 observations if available
    returns_df = returns.tail(60) if len(returns) > 60 else returns
    symbols = list(returns_df.columns)

    # Early exit for insufficient data
    if len(symbols) < 2 or len(returns_df) < 3:
        return {}

    # Union‑find initialization
    parent: Dict[str, str] = {s: s for s in symbols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    # Compute correlation matrix
    try:
        corr_matrix = returns_df.corr()
    except Exception as exc:
        logger.exception("Failed to compute correlation matrix", exc_info=exc)
        raise RuntimeError("Correlation computation failed") from exc

    # Build clusters based on threshold
    for i, s_a in enumerate(symbols):
        for s_b in symbols[i + 1 :]:
            try:
                corr = corr_matrix.loc[s_a, s_b]
                if abs(corr) > threshold:
                    union(s_a, s_b)
            except KeyError as ke:
                logger.warning(
                    "Missing correlation entry",
                    symbol_a=s_a,
                    symbol_b=s_b,
                    error=str(ke),
                )
                continue
            except TypeError as te:
                logger.warning(
                    "Non‑numeric correlation value",
                    symbol_a=s_a,
                    symbol_b=s_b,
                    error=str(te),
                )
                continue
            except Exception as exc:  # pragma: no cover
                logger.exception(
                    "Unexpected error during clustering",
                    symbol_a=s_a,
                    symbol_b=s_b,
                    exc_info=exc,
                )
                continue

    # Aggregate clusters
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
    """Return (allowed, reason). Blocks if adding ``new_symbol`` exceeds ``max_cluster_pct`` of equity.

    Parameters
    ----------
    new_symbol : str
        Symbol being added.
    new_value_usd : float
        Dollar value of the new position.
    current_positions : dict[str, float]
        Existing positions keyed by symbol.
    clusters : dict[str, list[str]]
        Mapping of cluster identifiers to member symbols.
    max_cluster_pct : float, optional
        Maximum allowed proportion of equity per cluster, by default 0.30.
    total_equity : float, optional
        Total portfolio equity, by default 100_000.

    Returns
    -------
    tuple[bool, str]
        ``(allowed, reason)`` where ``allowed`` is ``False`` when the limit would be breached.
    """
    # Input validation
    if not isinstance(new_symbol, str):
        logger.error(
            "Invalid new_symbol type",
            expected_type="str",
            actual_type=type(new_symbol).__name__,
        )
        raise TypeError("new_symbol must be a string")
    if not isinstance(new_value_usd, (float, int)):
        logger.error(
            "Invalid new_value_usd type",
            expected_type="numeric",
            actual_type=type(new_value_usd).__name__,
        )
        raise TypeError("new_value_usd must be a numeric type")
    if not isinstance(current_positions, dict):
        logger.error(
            "Invalid current_positions type",
            expected_type="dict",
            actual_type=type(current_positions).__name__,
        )
        raise TypeError("current_positions must be a dict")
    if not isinstance(clusters, dict):
        logger.error(
            "Invalid clusters type",
            expected_type="dict",
            actual_type=type(clusters).__name__,
        )
        raise TypeError("clusters must be a dict")
    if total_equity <= 0:
        logger.error(
            "Non‑positive total_equity",
            total_equity=total_equity,
        )
        raise ValueError("total_equity must be greater than zero")

    # Evaluate each cluster containing the new symbol
    for cluster_id, members in clusters.items():
        if new_symbol not in members:
            continue
        try:
            cluster_value = sum(current_positions.get(m, 0.0) for m in members) + new_value_usd
            cluster_pct = cluster_value / total_equity
        except Exception as exc:  # pragma: no cover
            logger.exception(
                "Error computing cluster limits",
                cluster_id=cluster_id,
                members=members,
                exc_info=exc,
            )
            raise RuntimeError("Failed to compute cluster limits") from exc

        if cluster_pct > max_cluster_pct:
            reason = (
                f"{new_symbol} would push {cluster_id} to {cluster_pct:.1%} "
                f"(max {max_cluster_pct:.1%})"
            )
            logger.warning(
                "Cluster limit breached",
                cluster=cluster_id,
                pct=cluster_pct,
                reason=reason,
            )
            return False, reason

    return True, "ok"