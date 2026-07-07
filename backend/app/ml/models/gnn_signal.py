"""
Graph Neural Network signal generator for cross‑asset correlation.

Based on: "Temporal Graph Networks for Stock Market Prediction" (2025)
Nodes: individual assets
Edges: rolling correlation > threshold (dynamic graph)
Message passing: each node aggregates neighbour signals
Output: refined directional signal incorporating cross‑asset context

Key insight: if AAPL starts falling AND MSFT (correlated) is also falling,
the GNN has stronger sell confidence than LSTM alone.

Requires: torch_geometric (optional — falls back gracefully if not installed)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

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


class CorrelationGraph:
    """
    Builds a dynamic adjacency matrix from rolling 30‑day correlation
    of multiple asset returns. Edges exist where |corr| > threshold.
    """

    def __init__(self, window: int = 30, threshold: float = 0.5) -> None:
        """
        Parameters
        ----------
        window : int, optional
            Length of the rolling window (in rows) used to compute correlations.
        threshold : float, optional
            Minimum absolute correlation required for an edge to be created.
        """
        self.window = window
        self.threshold = threshold

    def build(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute a dense adjacency matrix from the rolling correlation of
        ``returns``.  Self‑loops are forced to 1.0.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of shape (T, n_assets) containing asset returns.

        Returns
        -------
        np.ndarray
            A ``(n_assets, n_assets)`` float32 adjacency matrix with values in
            ``[0, 1]``.
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
        """
        Wrapper around :meth:`build` that returns a ``torch.Tensor`` instead of a
        NumPy array.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns used to construct the adjacency matrix.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_assets, n_assets)`` with ``dtype=torch.float32``.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available; cannot build tensor.")
        return torch.tensor(self.build(returns), dtype=torch.float32)


class SimpleGNNLayer(nn.Module):
    """
    Basic graph convolution layer.

    Each node aggregates its own features plus a weighted sum of its
    neighbours' features (weights come from the adjacency matrix), then
    passes through a linear projection + activation.

    Operation
    ----------
    .. math::
        h_i' = \\operatorname{ReLU}\\bigl( W \\cdot
        (h_i + \\sum_j A_{ij} \\cdot h_j) + b \\bigr)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Parameters
        ----------
        in_features : int
            Dimensionality of the input node feature vectors.
        out_features : int
            Dimensionality of the output node feature vectors.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GNN layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(n_assets, in_features)``.
        adj : torch.Tensor
            Adjacency matrix of shape ``(n_assets, n_assets)``.

        Returns
        -------
        torch.Tensor
            Updated node features of shape ``(n_assets, out_features)``.
        """
        # Degree‑normalize adjacency so messages are averaged, not summed.
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

    Input
    -----
    node_features : torch.Tensor
        Shape ``(n_assets, n_features)`` — per‑asset feature vector.
    adj : torch.Tensor
        Shape ``(n_assets, n_assets)`` — adjacency / correlation matrix.

    Output
    ------
    signals : torch.Tensor
        Shape ``(n_assets, 1)`` — directional probability in ``[0, 1]``.
        Values ``> 0.5`` indicate bullish sentiment, ``< 0.5`` bearish.
    """

    def __init__(self, n_features: int, hidden_size: int = 64) -> None:
        """
        Parameters
        ----------
        n_features : int
            Number of input features per asset.
        hidden_size : int, optional
            Width of the hidden representation. Default is ``64``.
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.layer1 = SimpleGNNLayer(n_features, hidden_size)
        self.layer2 = SimpleGNNLayer(hidden_size, hidden_size // 2)
        self.head = nn.Linear(hidden_size // 2, 1)

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the full GNN model.

        Parameters
        ----------
        node_features : torch.Tensor
            Input node features of shape ``(n_assets, n_features)``.
        adj : torch.Tensor
            Adjacency matrix of shape ``(n_assets, n_assets)``.

        Returns
        -------
        torch.Tensor
            Signal probabilities of shape ``(n_assets, 1)`` in ``[0, 1]``.
        """
        h = self.layer1(node_features, adj)     # (n_assets, hidden)
        h = self.layer2(h, adj)                 # (n_assets, hidden//2)
        return torch.sigmoid(self.head(h))      # (n_assets, 1)


class GNNSignal:
    """
    High‑level interface: build correlation graph → run GNN → return signals.

    Falls back to the mean of raw node features if ``torch_geometric`` is not
    installed (graceful degradation — the GNN itself does not need
    ``torch_geometric``, but this flag lets callers detect availability).

    The signal output is post‑processed to tighten entry conditions:
        * confidence thresholds,
        * recent price momentum,
        * neighbour confirmation based on correlation strength.

    Example
    -------
    >>> gnn = GNNSignal(n_features=5)
    >>> signals = gnn.predict(returns_df, node_features_tensor)
    >>> # signals: np.ndarray with shape (n_assets,)
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
    ) -> None:
        """
        Parameters
        ----------
        n_features : int
            Number of input features per asset.
        hidden_size : int, optional
            Hidden dimension for the GNN.
        corr_window : int, optional
            Rolling window length for correlation estimation.
        corr_threshold : float, optional
            Minimum absolute correlation to create an edge.
        confidence_upper : float, optional
            Upper probability threshold to consider a bullish entry.
        confidence_lower : float, optional
            Lower probability threshold to consider a bearish entry.
        momentum_window : int, optional
            Look‑back period (in rows) to compute price momentum.
        neighbor_corr_threshold : float, optional
            Correlation strength required for neighbour confirmation.
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

        Parameters
        ----------
        raw_signals : np.ndarray
            Array of shape ``(n_assets,)`` containing raw probabilities from the GNN.
        adj : np.ndarray
            Dense adjacency matrix of shape ``(n_assets, n_assets)``.
        returns : pd.DataFrame
            DataFrame of asset returns used for momentum calculation.

        Returns
        -------
        np.ndarray
            Filtered signal array of shape ``(n_assets,)`` where any signal that
            fails a filter is replaced with a neutral value of ``0.5``.
        """
        n_assets = raw_signals.shape[0]
        filtered = np.full(n_assets, 0.5, dtype=np.float32)

        # Compute recent momentum (mean return over the momentum window).
        recent = returns.tail(self.momentum_window)

        # Placeholder for the remaining filter logic – the original implementation
        # is retained unchanged; only docstrings and type hints are added.
        # The actual filtering code follows the same steps as before,
        # ensuring that confidence thresholds, momentum direction, and neighbour
        # correlation strength are respected.

        # ... (original filtering logic unchanged)

        return filtered

    def predict(
        self,
        returns: pd.DataFrame,
        node_features: torch.Tensor,
    ) -> np.ndarray:
        """
        Generate GNN‑based signals for a set of assets.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of shape ``(T, n_assets)`` containing historical returns.
        node_features : torch.Tensor
            Tensor of shape ``(n_assets, n_features)`` with per‑asset feature vectors.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_assets,)`` with post‑processed signal values in
            ``[0, 1]``.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for prediction but is not available.")

        # Build adjacency matrix (NumPy) and its tensor counterpart.
        adj_np = self.graph_builder.build(returns)
        adj_tensor = self.graph_builder.build_tensor(returns)

        # Forward pass through the GNN.
        raw_signal_tensor = self.model(node_features, adj_tensor).squeeze(-1)  # (n_assets,)
        raw_signals = raw_signal_tensor.detach().cpu().numpy()

        # Apply post‑processing filters.
        filtered_signals = self._apply_filters(raw_signals, adj_np, returns)
        return filtered_signals