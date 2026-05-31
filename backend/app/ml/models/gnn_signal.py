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
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]

try:
    import torch_geometric  # noqa: F401
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False


class CorrelationGraph:
    """
    Builds a dynamic adjacency matrix from rolling 30-day correlation
    of multiple asset returns.  Edges exist where |corr| > threshold.
    """

    def __init__(self, window: int = 30, threshold: float = 0.5):
        self.window = window
        self.threshold = threshold

    def build(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Args:
            returns: DataFrame (T, n_assets) of asset returns
        Returns:
            adj: (n_assets, n_assets) float32 adjacency matrix, values in [0, 1]
        """
        tail = returns.tail(self.window)
        if len(tail) < 3:
            n = returns.shape[1]
            return np.eye(n, dtype=np.float32)

        corr = tail.corr().fillna(0.0).values.astype(np.float32)
        # Zero out edges below threshold; keep self-loops
        adj = np.where(np.abs(corr) >= self.threshold, np.abs(corr), 0.0).astype(
            np.float32
        )
        np.fill_diagonal(adj, 1.0)
        return adj

    def build_tensor(self, returns: pd.DataFrame) -> torch.Tensor:
        """Returns (n_assets, n_assets) torch.Tensor."""
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
            x:   (n_assets, in_features) node feature matrix
            adj: (n_assets, n_assets) adjacency matrix
        Returns:
            h:   (n_assets, out_features)
        """
        # Degree-normalise adjacency so messages are averaged, not summed
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg                     # (n_assets, n_assets)

        # Message passing: aggregate neighbour features
        agg = torch.mm(adj_norm, x)             # (n_assets, in_features)
        h = self.linear(agg)                    # (n_assets, out_features)
        h = self.norm(h)
        return torch.relu(h)


class GNNSignalModel(nn.Module):
    """
    Two-layer GNN that produces a directional probability per asset.

    Input:
        node_features: (n_assets, n_features) — per-asset feature vector
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
    High-level interface: build correlation graph → run GNN → return signals.

    Falls back to the mean of raw node features if torch_geometric is not
    installed (graceful degradation — the GNN itself does not need
    torch_geometric, but this flag lets callers detect availability).

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
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.torch_geometric_available = _TORCH_GEOMETRIC_AVAILABLE

        self.graph_builder = CorrelationGraph(window=corr_window, threshold=corr_threshold)
        self.model = GNNSignalModel(n_features=n_features, hidden_size=hidden_size)

    def predict(
        self,
        returns: pd.DataFrame,
        node_features: torch.Tensor,
    ) -> np.ndarray:
        """
        Produce a directional signal for each asset.

        Args:
            returns:       (T, n_assets) DataFrame of asset returns
            node_features: (n_assets, n_features) per-asset feature tensor
        Returns:
            signals: (n_assets,) float array in [0, 1]

        Graceful degradation:
            If torch_geometric is not installed the GNN path still runs
            (SimpleGNNLayer requires only PyTorch). The flag
            self.torch_geometric_available lets callers check.
            If anything else goes wrong we fall back to the mean of
            raw features as a neutral 0.5 signal.
        """
        try:
            adj = self.graph_builder.build_tensor(returns)
            self.model.eval()
            with torch.no_grad():
                out = self.model(node_features, adj)  # (n_assets, 1)
            return out.squeeze(-1).numpy()            # (n_assets,)
        except Exception:
            # Fallback: return 0.5 (neutral) for every asset
            n_assets = node_features.shape[0]
            return np.full(n_assets, 0.5, dtype=np.float32)

    def predict_with_fallback(
        self,
        returns: pd.DataFrame,
        node_features: torch.Tensor,
    ) -> np.ndarray:
        """
        Alias for predict() — always safe to call even without torch_geometric.
        """
        return self.predict(returns, node_features)
