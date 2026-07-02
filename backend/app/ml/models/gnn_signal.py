"""
Graph Neural Network signal generator for cross-asset correlation.

Based on: "Temporal Graph Networks for Stock Market Prediction" (2025)
Nodes: individual assets
Edges: rolling correlation > threshold (dynamic graph)
Message passing: each node aggregates neighbour signals
Output: refined directional signal incorporating cross-asset context

Key insight: if AAPL starts falling AND MSFT (correlated) is also falling,
the GNN has stronger sell confidence than LSTM alone.

Requires: torch_geometric (optional — falls back gracefully if not installed)
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]

try:
    import torch_geometric  # noqa: F401
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_GEOMETRIC_AVAILABLE = False

# Configure module‑level logger
_logger = logging.getLogger(__name__)


class CorrelationGraph:
    """
    Builds a dynamic adjacency matrix from rolling 30‑day correlation
    of multiple asset returns. Edges exist where |corr| > threshold.
    """

    def __init__(self, window: int = 30, threshold: float = 0.5):
        self.window = window
        self.threshold = threshold

    def build(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Args:
            returns: DataFrame (T, n_assets) of asset returns.

        Returns:
            adj: (n_assets, n_assets) float32 adjacency matrix, values in [0, 1].
        """
        tail = returns.tail(self.window)
        if len(tail) < 3:
            n = returns.shape[1]
            return np.eye(n, dtype=np.float32)

        corr = tail.corr().fillna(0.0).values.astype(np.float32)
        # Keep absolute correlation where it exceeds the threshold; otherwise zero.
        adj = np.where(np.abs(corr) >= self.threshold, np.abs(corr), 0.0).astype(np.float32)
        np.fill_diagonal(adj, 1.0)  # ensure self‑loops
        return adj

    def build_tensor(self, returns: pd.DataFrame) -> torch.Tensor:
        """Returns (n_assets, n_assets) torch.Tensor."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available; cannot build tensor.")
        return torch.tensor(self.build(returns), dtype=torch.float32)


class SimpleGNNLayer(nn.Module):
    """
    Basic graph convolution layer.

    Each node aggregates its own features plus a weighted sum of its
    neighbours' features (weights come from the adjacency matrix), then
    passes through a linear projection + activation.

    Operation:
        h_i' = ReLU( W · (h_i + Σ_j A_ij · h_j) + b )
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (n_assets, in_features) node feature matrix.
            adj: (n_assets, n_assets) adjacency matrix.

        Returns:
            h:   (n_assets, out_features)
        """
        # Degree‑normalise adjacency so messages are averaged, not summed.
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg  # (n_assets, n_assets)

        # Message passing: aggregate neighbour features.
        agg = torch.mm(adj_norm, x)  # (n_assets, in_features)
        h = self.linear(agg)        # (n_assets, out_features)
        h = self.norm(h)
        return torch.relu(h)


class GNNSignalModel(nn.Module):
    """
    Two‑layer GNN that produces a directional probability per asset.

    Input:
        node_features: (n_assets, n_features) — per‑asset feature vector
        adj:           (n_assets, n_assets)   — adjacency / correlation matrix

    Output:
        signals: (n_assets, 1) — directional probability in [0, 1]
                 > 0.5 → bullish, < 0.5 → bearish
    """

    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.layer1 = SimpleGNNLayer(n_features, hidden_size)
        self.layer2 = SimpleGNNLayer(hidden_size, hidden_size // 2)
        self.head = nn.Linear(hidden_size // 2, 1)

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (n_assets, n_features)
            adj:           (n_assets, n_assets)

        Returns:
            signals: (n_assets, 1) in [0, 1]
        """
        h = self.layer1(node_features, adj)     # (n_assets, hidden)
        h = self.layer2(h, adj)                 # (n_assets, hidden//2)
        return torch.sigmoid(self.head(h))      # (n_assets, 1)


class GNNSignal:
    """
    High‑level interface: build correlation graph → run GNN → return signals.

    Falls back to the mean of raw node features if torch_geometric is not
    installed (graceful degradation — the GNN itself does not need
    torch_geometric, but this flag lets callers detect availability).

    The signal output is post‑processed to tighten entry conditions:
        * confidence thresholds,
        * recent price momentum,
        * neighbour confirmation based on correlation strength.

    Usage:
        gnn = GNNSignal(n_features=5)
        signals = gnn.predict(returns_df, node_features_tensor)
        # signals: np.ndarray (n_assets,)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        corr_window: int = 30,
        corr_threshold: float = 0.5,
        *,
        confidence_upper: float = 0.6,
        confidence_lower: float = 0.4,
        momentum_window: int = 5,
        neighbor_corr_threshold: float = 0.7,
    ):
        """
        Args:
            n_features: Number of input features per asset.
            hidden_size: Hidden dimension for the GNN.
            corr_window: Rolling window length for correlation estimation.
            corr_threshold: Minimum absolute correlation to create an edge.
            confidence_upper: Upper probability threshold to consider a bullish entry.
            confidence_lower: Lower probability threshold to consider a bearish entry.
            momentum_window: Look‑back period (in rows) to compute price momentum.
            neighbor_corr_threshold: Correlation strength required for neighbour confirmation.
        """
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.torch_geometric_available = _TORCH_GEOMETRIC_AVAILABLE

        self.graph_builder = CorrelationGraph(window=corr_window, threshold=corr_threshold)
        self.model = GNNSignalModel(n_features=n_features, hidden_size=hidden_size)

        # Filtering parameters
        self.confidence_upper = confidence_upper
        self.confidence_lower = confidence_lower
        self.momentum_window = momentum_window
        self.neighbor_corr_threshold = neighbor_corr_threshold

    def _apply_filters(
        self,
        raw_signals: np.ndarray,
        adj: np.ndarray,
        returns: pd.DataFrame,
    ) -> np.ndarray:
        """
        Tighten entry conditions using confidence thresholds, recent momentum,
        and neighbour confirmation.

        Args:
            raw_signals: (n_assets,) raw probabilities from the GNN.
            adj: (n_assets, n_assets) adjacency matrix (numpy).
            returns: DataFrame of asset returns.

        Returns:
            filtered_signals: (n_assets,) array where signals failing any filter
                              are set to a neutral 0.5 value.
        """
        n_assets = raw_signals.shape[0]
        filtered = np.full(n_assets, 0.5, dtype=np.float32)

        # Compute recent momentum (mean return over the momentum window).
        recent = returns.tail(self.momentum_window)

        # Confidence filter
        bullish = raw_signals >= self.confidence_upper
        bearish = raw_signals <= self.confidence_lower
        confidence_mask = bullish | bearish

        # Momentum filter: require sign of recent mean return to match signal direction.
        momentum = recent.mean().values  # shape (n_assets,)
        momentum_mask = np.where(bullish, momentum > 0, np.where(bearish, momentum < 0, False))

        # Neighbour confirmation: at least one neighbour with correlation > threshold
        # shares the same signal direction.
        neighbour_mask = np.zeros(n_assets, dtype=bool)
        for i in range(n_assets):
            if not confidence_mask[i]:
                continue
            # Identify neighbours with strong correlation.
            neighbours = np.where(adj[i] >= self.neighbor_corr_threshold)[0]
            if neighbours.size == 0:
                continue
            # Compare neighbour signals (using raw_signals).
            same_dir = ((raw_signals[neighbours] >= 0.5) & bullish[i]) | \
                       ((raw_signals[neighbours] < 0.5) & bearish[i])
            neighbour_mask[i] = same_dir.any()

        # Combine all masks.
        final_mask = confidence_mask & momentum_mask & neighbour_mask
        filtered[final_mask] = raw_signals[final_mask]
        return filtered

    def predict(self, returns: pd.DataFrame, node_features: torch.Tensor) -> np.ndarray:
        """
        Generate GNN‑based signals, apply post‑filters, and emit structured logs.

        Args:
            returns: DataFrame of asset returns (T, n_assets).
            node_features: Tensor of shape (n_assets, n_features).

        Returns:
            np.ndarray of shape (n_assets,) containing filtered signals.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GNNSignal prediction.")

        start_time = time.time()

        # Build adjacency matrix.
        adj_np = self.graph_builder.build(returns)
        adj_tensor = torch.tensor(adj_np, dtype=torch.float32, device=node_features.device)

        # Forward pass through the GNN.
        raw_tensor = self.model(node_features, adj_tensor)
        raw_np = raw_tensor.detach().cpu().numpy().reshape(-1)

        # Apply domain‑specific filters.
        filtered_np = self._apply_filters(raw_np, adj_np, returns)

        # Metrics for monitoring.
        signal_count = int(((filtered_np > 0.5) | (filtered_np < 0.5)).sum())
        exec_time = time.time() - start_time

        # Simple P&L estimate: (signal - 0.5) * latest return summed across assets.
        # This is a lightweight proxy for monitoring; real P&L is computed downstream.
        latest_return = returns.tail(1).values.squeeze()
        if latest_return.shape[0] != filtered_np.shape[0]:
            # Defensive fallback if shapes mismatch.
            estimated_pnl = 0.0
        else:
            estimated_pnl = float(((filtered_np - 0.5) * latest_return).sum())

        _logger.info(
            "GNNSignal prediction completed",
            extra={
                "signal_count": signal_count,
                "execution_time_s": exec_time,
                "estimated_pnl": estimated_pnl,
            },
        )

        return filtered_np