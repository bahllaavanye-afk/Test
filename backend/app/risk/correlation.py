"""Detect correlated clusters and enforce allocation limits per cluster."""
import numpy as np
import pandas as pd
from app.utils.logging import logger


def compute_correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """Union-find connected components for correlation clustering."""
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
        for s_b in symbols[i + 1:]:
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
    for cluster_id, members in clusters.items():
        if new_symbol not in members:
            continue
        cluster_value = sum(current_positions.get(m, 0.0) for m in members) + new_value_usd
        cluster_pct = cluster_value / total_equity
        if cluster_pct > max_cluster_pct:
            reason = f"{new_symbol} would push {cluster_id} to {cluster_pct:.1%} (max {max_cluster_pct:.1%})"
            logger.warning("Cluster limit breached", **{"cluster": cluster_id, "pct": cluster_pct})
            return False, reason
    return True, "ok"


# ----------------------------------------------------------------------
# Unit tests for edge and boundary conditions
# ----------------------------------------------------------------------
import unittest


class TestCorrelationClusters(unittest.TestCase):
    def test_empty_dataframe_returns_empty_clusters(self):
        empty_df = pd.DataFrame()
        result = compute_correlation_clusters(empty_df)
        self.assertEqual(result, {})

    def test_insufficient_rows_or_symbols_returns_empty(self):
        # One symbol with sufficient rows
        df_one_symbol = pd.DataFrame(
            {"A": np.random.randn(5)},
        )
        self.assertEqual(compute_correlation_clusters(df_one_symbol), {})

        # Multiple symbols but less than three rows
        df_few_rows = pd.DataFrame(
            {"A": [0.1, 0.2], "B": [0.2, 0.3]},
        )
        self.assertEqual(compute_correlation_clusters(df_few_rows), {})

    def test_correlation_at_threshold_is_not_clustered(self):
        # Create two perfectly correlated series with correlation exactly 0.70
        np.random.seed(0)
        base = np.random.randn(10)
        a = base
        b = base * 0.7 + np.random.randn(10) * np.sqrt(1 - 0.7**2)  # correlation ~0.7
        df = pd.DataFrame({"A": a, "B": b})
        # Force correlation to be exactly the threshold
        corr = df.corr().loc["A", "B"]
        self.assertAlmostEqual(abs(corr), 0.70, places=2)

        clusters = compute_correlation_clusters(df, threshold=0.70)
        # Expect each symbol to be its own cluster because condition uses '>'
        self.assertEqual(set(clusters.keys()), {"A", "B"})
        for members in clusters.values():
            self.assertEqual(len(members), 1)

    def test_cluster_limit_boundary_allows_exact_max(self):
        clusters = {"cluster1": ["X", "Y"]}
        current_positions = {"X": 20_000, "Y": 10_000}
        # Adding new symbol Z that belongs to the same cluster would hit exactly 30%
        allowed, reason = check_cluster_limits(
            new_symbol="Z",
            new_value_usd=0,  # No additional value, just testing existing cluster
            current_positions=current_positions,
            clusters=clusters,
            max_cluster_pct=0.30,
            total_equity=100_000,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "ok")

        # Adding value that pushes just over the limit should be blocked
        allowed, reason = check_cluster_limits(
            new_symbol="Z",
            new_value_usd=1,  # 1 USD over the limit
            current_positions=current_positions,
            clusters=clusters,
            max_cluster_pct=0.30,
            total_equity=100_000,
        )
        self.assertFalse(allowed)
        self.assertIn("would push", reason)


if __name__ == "__main__":
    unittest.main()