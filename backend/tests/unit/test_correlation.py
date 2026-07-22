"""Correlation cluster unit tests.

These tests verify the behavior of the correlation clustering utilities used in
risk management. The focus is on ensuring that clusters are correctly formed
based on a correlation threshold and that the cluster exposure limits enforce
the intended risk caps.
"""

import pandas as pd
import numpy as np
from typing import Callable

from app.risk.correlation import compute_correlation_clusters, check_cluster_limits


def _make_returns(corr_matrix: np.ndarray, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic return series with a prescribed correlation matrix.

    Args:
        corr_matrix: Target correlation matrix (must be positive‑definite).
        n: Number of observations to simulate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of shape (n, n_assets) with column names ``S0``, ``S1`` ….
    """
    rng = np.random.default_rng(seed)
    n_assets = corr_matrix.shape[0]
    # Cholesky factor produces the required covariance structure.
    L = np.linalg.cholesky(corr_matrix)
    Z = rng.standard_normal((n, n_assets))
    returns = Z @ L.T
    return pd.DataFrame(returns, columns=[f"S{i}" for i in range(n_assets)])


def test_compute_clusters_independent_assets() -> None:
    """Assets with zero correlation should each form their own cluster."""
    corr = np.eye(3)
    returns = _make_returns(corr)
    clusters = compute_correlation_clusters(returns, threshold=0.70)
    # Expect three singleton clusters.
    assert isinstance(clusters, dict)
    assert len(clusters) == 3
    for members in clusters.values():
        assert len(members) == 1


def test_compute_clusters_perfect_correlation() -> None:
    """Highly correlated assets must be grouped together."""
    corr = np.array(
        [
            [1.0, 0.99, 0.99],
            [0.99, 1.0, 0.99],
            [0.99, 0.99, 1.0],
        ]
    )
    returns = _make_returns(corr)
    clusters = compute_correlation_clusters(returns, threshold=0.70)
    # At least one cluster should contain all three members.
    assert any(len(members) >= 3 for members in clusters.values())


def test_compute_clusters_threshold_edge() -> None:
    """A correlation exactly at the threshold should be considered linked."""
    corr = np.array(
        [
            [1.0, 0.70, 0.30],
            [0.70, 1.0, 0.30],
            [0.30, 0.30, 1.0],
        ]
    )
    returns = _make_returns(corr)
    clusters = compute_correlation_clusters(returns, threshold=0.70)
    # Assets 0 and 1 must be clustered together, asset 2 separate.
    cluster_sizes = sorted(len(m) for m in clusters.values())
    assert cluster_sizes == [1, 2]


def test_check_cluster_limits_blocks() -> None:
    """Exposure beyond the allowed cluster percentage must be rejected."""
    clusters = {"cluster_0": ["AAPL", "MSFT", "GOOGL"]}
    positions = {"AAPL": 20_000, "MSFT": 15_000}
    allowed, reason = check_cluster_limits(
        "GOOGL",
        new_value_usd=20_000,
        current_positions=positions,
        clusters=clusters,
        max_cluster_pct=0.30,
        total_equity=100_000,
    )
    # 55 % > 30 % → not allowed.
    assert allowed is False
    assert "cluster" in reason.lower()
    assert "exceed" in reason.lower()


def test_check_cluster_limits_allows() -> None:
    """Exposure within the cluster limit should be permitted."""
    clusters = {"cluster_0": ["AAPL", "MSFT"]}
    positions = {"AAPL": 5_000}
    allowed, reason = check_cluster_limits(
        "MSFT",
        new_value_usd=5_000,
        current_positions=positions,
        clusters=clusters,
        max_cluster_pct=0.30,
        total_equity=100_000,
    )
    assert allowed is True
    assert reason == ""


def test_unknown_symbol_allowed() -> None:
    """Symbols not present in any cluster are always allowed."""
    clusters = {"cluster_0": ["AAPL"]}
    allowed, reason = check_cluster_limits("NVDA", 10_000, {}, clusters, 0.30, 100_000)
    assert allowed is True
    assert reason == ""