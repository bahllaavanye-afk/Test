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
    nn = None  # type: ignore[assignment]

try:
    import torch_geometric  # noqa: F401
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_GEOMETRIC_AVAILABLE = False

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
        if recent.empty:
            momentum = np.zeros(n_assets, dtype=np.float32)
        else:
            momentum = recent.mean().values.astype(np.float32)

        # Neighbour confirmation: for each asset, check max correlation with neighbours.
        max_neighbor_corr = adj.max(axis=1).astype(np.float32)

        for i in range(n_assets):
            signal = raw_signals[i]
            # Confidence thresholds.
            if signal >= self.confidence_upper and momentum[i] > 0 and max_neighbor_corr[i] >= self.neighbor_corr_threshold:
                filtered[i] = signal
            elif signal <= self.confidence_lower and momentum[i] < 0 and max_neighbor_corr[i] >= self.neighbor_corr_threshold:
                filtered[i] = signal
            else:
                filtered[i] = 0.5  # neutral

        return filtered

    def predict(self, returns: pd.DataFrame, node_features: torch.Tensor) -> np.ndarray:
        """
        Generate filtered GNN signals for the given returns and node features.

        Args:
            returns: DataFrame (T, n_assets) of asset returns.
            node_features: Tensor (n_assets, n_features) of per‑asset features.

        Returns:
            np.ndarray of shape (n_assets,) containing filtered signals.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available; cannot run GNNSignal prediction.")

        start_time = time.time()

        # Build adjacency matrix.
        adj_np = self.graph_builder.build(returns)
        adj_tensor = torch.tensor(adj_np, dtype=torch.float32)

        # Forward pass through the GNN.
        raw_tensor = self.model(node_features, adj_tensor)  # shape (n_assets, 1)
        raw_signals = raw_tensor.squeeze(-1).detach().cpu().numpy().astype(np.float32)

        # Apply post‑processing filters.
        filtered_signals = self._apply_filters(raw_signals, adj_np, returns)

        exec_time = time.time() - start_time
        signal_count = int(np.sum(filtered_signals > 0.5))

        # Simple P&L estimate: sum of last returns weighted by bullish signals.
        last_returns = returns.iloc[-1].values.astype(np.float32)
        pnl_estimate = float(np.dot((filtered_signals > 0.5).astype(np.float32), last_returns))

        _logger.info(
            "GNNSignal.predict executed in %.3fs | signal_count=%d | pnl_estimate=%.4f",
            exec_time,
            signal_count,
            pnl_estimate,
        )

        return filtered_signals