"""Correlation cluster tests."""
import pandas as pd
import numpy as np
from app.risk.correlation import compute_correlation_clusters, check_cluster_limits


def _make_returns(corr_matrix, n=100, seed=42):
    """Generate synthetic returns with given correlation structure."""
    rng = np.random.default_rng(seed)
    n_assets = corr_matrix.shape[0]
    L = np.linalg.cholesky(corr_matrix)
    Z = rng.standard_normal((n, n_assets))
    returns = Z @ L.T
    return pd.DataFrame(returns, columns=[f"S{i}" for i in range(n_assets)])


def test_compute_clusters_independent_assets():
    corr = np.eye(3)
    returns = _make_returns(corr)
    clusters = compute_correlation_clusters(returns, threshold=0.70)
    # 3 assets with no correlation → 3 clusters
    assert len(clusters) == 3


def test_compute_clusters_perfect_correlation():
    corr = np.array([[1.0, 0.99, 0.99],
                     [0.99, 1.0, 0.99],
                     [0.99, 0.99, 1.0]])
    returns = _make_returns(corr)
    clusters = compute_correlation_clusters(returns, threshold=0.70)
    # All three should be in one cluster
    assert any(len(members) >= 2 for members in clusters.values())


def test_check_cluster_limits_blocks():
    clusters = {"cluster_0": ["AAPL", "MSFT", "GOOGL"]}
    positions = {"AAPL": 20_000, "MSFT": 15_000}
    allowed, reason = check_cluster_limits(
        "GOOGL", new_value_usd=20_000, current_positions=positions,
        clusters=clusters, max_cluster_pct=0.30, total_equity=100_000,
    )
    # 20k + 15k + 20k = 55k = 55% > 30%
    assert not allowed
    assert "cluster" in reason.lower()


def test_check_cluster_limits_allows():
    clusters = {"cluster_0": ["AAPL", "MSFT"]}
    positions = {"AAPL": 5_000}
    allowed, _ = check_cluster_limits(
        "MSFT", new_value_usd=5_000, current_positions=positions,
        clusters=clusters, max_cluster_pct=0.30, total_equity=100_000,
    )
    # 5k + 5k = 10k = 10% < 30%
    assert allowed


def test_unknown_symbol_allowed():
    clusters = {"cluster_0": ["AAPL"]}
    allowed, _ = check_cluster_limits("NVDA", 10_000, {}, clusters, 0.30, 100_000)
    assert allowed
