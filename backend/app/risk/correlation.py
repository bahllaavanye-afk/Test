"""Detect correlated clusters and enforce allocation limits per cluster."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field, confloat, validator

from app.utils.logging import logger


class CorrelationClusterRequest(BaseModel):
    """Request model for computing correlation clusters.

    Attributes
    ----------
    returns : List[Dict[str, float]]
        Historical return series for each symbol. Each dict represents a row
        where keys are ticker symbols and values are the corresponding returns.
        The list should contain at least three rows and at least two symbols.
    threshold : float, optional
        Correlation magnitude threshold used to decide if two symbols belong
        to the same cluster. Must be between 0 and 1 (inclusive). Default is
        ``0.70``.
    """

    returns: List[Dict[str, float]] = Field(
        ...,
        description="Historical returns for each symbol, ordered chronologically.",
        example=[
            {"AAPL": 0.001, "MSFT": -0.002, "GOOG": 0.0005},
            {"AAPL": -0.001, "MSFT": 0.003, "GOOG": -0.001},
            {"AAPL": 0.002, "MSFT": 0.001, "GOOG": 0.0002},
        ],
    )
    threshold: confloat(ge=0.0, le=1.0) = Field(
        0.70,
        description="Absolute correlation threshold for clustering.",
        example=0.70,
    )

    @validator("returns")
    def validate_returns(cls, v: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if len(v) < 3:
            raise ValueError("At least three rows of returns are required.")
        # Ensure each row has the same set of symbols
        symbols = set(v[0].keys())
        if len(symbols) < 2:
            raise ValueError("At least two symbols are required for clustering.")
        for row in v:
            if set(row.keys()) != symbols:
                raise ValueError("All rows must contain the same symbols.")
        return v


class CorrelationClusterResponse(BaseModel):
    """Response model containing the computed correlation clusters.

    Attributes
    ----------
    clusters : Dict[str, List[str]]
        Mapping from a cluster identifier (root symbol) to the list of symbols
        belonging to that cluster.
    """

    clusters: Dict[str, List[str]] = Field(
        ...,
        description="Dictionary of cluster roots to their member symbols.",
        example={"AAPL": ["AAPL", "MSFT"], "GOOG": ["GOOG"]},
    )


class ClusterLimitCheckRequest(BaseModel):
    """Request model for checking allocation limits within a correlation cluster.

    Attributes
    ----------
    new_symbol : str
        Symbol of the position being added.
    new_value_usd : float
        Dollar value of the new position.
    current_positions : Dict[str, float]
        Mapping of existing symbol positions to their dollar values.
    clusters : Dict[str, List[str]]
        Correlation clusters as returned by ``CorrelationClusterResponse``.
    max_cluster_pct : float, optional
        Maximum allowed proportion of total equity that a single cluster may
        occupy. Must be between 0 and 1. Default is ``0.30``.
    total_equity : float, optional
        Total equity against which cluster percentages are measured. Default is
        ``100_000``.
    """

    new_symbol: str = Field(
        ...,
        description="Symbol of the position to be added.",
        example="AAPL",
    )
    new_value_usd: confloat(gt=0) = Field(
        ...,
        description="Dollar amount of the new position.",
        example=15000.0,
    )
    current_positions: Dict[str, float] = Field(
        ...,
        description="Existing positions expressed as symbol‑to‑USD mapping.",
        example={"AAPL": 20000.0, "MSFT": 15000.0},
    )
    clusters: Dict[str, List[str]] = Field(
        ...,
        description="Correlation clusters mapping root symbol to member symbols.",
        example={"AAPL": ["AAPL", "MSFT"], "GOOG": ["GOOG"]},
    )
    max_cluster_pct: confloat(ge=0.0, le=1.0) = Field(
        0.30,
        description="Maximum allowed cluster exposure as a fraction of total equity.",
        example=0.30,
    )
    total_equity: confloat(gt=0) = Field(
        100_000,
        description="Total equity used for percentage calculations.",
        example=100_000,
    )

    @validator("clusters")
    def validate_clusters(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        if not v:
            raise ValueError("Clusters dictionary cannot be empty.")
        for root, members in v.items():
            if not members:
                raise ValueError(f"Cluster '{root}' must contain at least one member.")
        return v


class ClusterLimitCheckResponse(BaseModel):
    """Response model indicating whether a new position respects cluster limits.

    Attributes
    ----------
    allowed : bool
        ``True`` if the position complies with the cluster limit, otherwise ``False``.
    reason : str
        Human‑readable explanation for the decision.
    """

    allowed: bool = Field(
        ...,
        description="Whether the new position is allowed within cluster limits.",
        example=True,
    )
    reason: str = Field(
        ...,
        description="Explanation of the decision.",
        example="ok",
    )


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """Union‑find connected components for correlation clustering.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of historical returns where columns are symbols.
    threshold : float, optional
        Absolute correlation threshold used to join symbols into a cluster.

    Returns
    -------
    dict[str, list[str]]
        Mapping from cluster root symbol to the list of symbols in that cluster.
    """
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
    """Return (allowed, reason). Blocks if adding ``new_symbol`` exceeds ``max_cluster_pct`` of equity.

    Parameters
    ----------
    new_symbol : str
        Symbol of the position being considered.
    new_value_usd : float
        Dollar value of the new position.
    current_positions : dict[str, float]
        Existing positions mapped to their USD values.
    clusters : dict[str, list[str]]
        Correlation clusters returned by ``compute_correlation_clusters``.
    max_cluster_pct : float, optional
        Maximum allowed proportion of total equity for any single cluster.
    total_equity : float, optional
        Total portfolio equity used for percentage calculations.

    Returns
    -------
    tuple[bool, str]
        ``(allowed, reason)`` where ``allowed`` indicates compliance and ``reason``
        provides a human‑readable explanation.
    """
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


__all__ = [
    "CorrelationClusterRequest",
    "CorrelationClusterResponse",
    "ClusterLimitCheckRequest",
    "ClusterLimitCheckResponse",
    "compute_correlation_clusters",
    "check_cluster_limits",
]