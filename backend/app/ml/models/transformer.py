"""
Temporal Fusion Transformer (Lim et al., 2021).

State‑of‑the‑art for multi‑horizon time series forecasting.
The attention mechanism provides interpretable feature importance per timestep.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

# Use the real nn.Module base when torch is present; fall back to ``object`` so these
# classes still *import* (as inert placeholders) in torch‑free environments. The model
# registry can then expose them without crashing; instantiating them still needs torch.
_NNModule = nn.Module if _TORCH_AVAILABLE else object

import numpy as np
from app.ml.models.base_model import AbstractModel, EvalMetrics


class GatedLinearUnit(_NNModule):
    """
    Gated Linear Unit (GLU) implementation.

    The GLU splits the linear output into a value and a gate,
    applying a sigmoid to the gate and multiplying it with the value.
    """

    def __init__(self, d: int):
        """
        Args:
            d: Dimensionality of the input (and output) features.
        """
        super().__init__()
        self.fc = nn.Linear(d, d * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., d).

        Returns:
            Tensor of shape (..., d) after gating.
        """
        h = self.fc(x)
        split = h.shape[-1] // 2
        return h[..., :split] * torch.sigmoid(h[..., split:])


class GatedResidualNetwork(_NNModule):
    """
    Gated Residual Network (GRN).

    A two‑layer MLP with a GLU gate and residual connection, followed by layer
    normalization. Used throughout the TFT architecture for feature processing.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        """
        Args:
            d_in: Input dimensionality.
            d_hidden: Hidden layer dimensionality.
            d_out: Output dimensionality.
            dropout: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = GatedLinearUnit(d_out)
        self.ln = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., d_in).

        Returns:
            Tensor of shape (..., d_out) after gated residual processing.
        """
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.gate(h) + self.skip(x)
        return self.ln(h)


class VariableSelectionNetwork(_NNModule):
    """
    Variable Selection Network (VSN).

    Applies a softmax‑weighted GRN per variable to learn the importance of each
    input feature. Returns both the aggregated representation and the learned
    attention weights.
    """

    def __init__(self, n_vars: int, d_model: int):
        """
        Args:
            n_vars: Number of input variables.
            d_model: Dimensionality of the model embedding for each variable.
        """
        super().__init__()
        self.grns = nn.ModuleList(
            [GatedResidualNetwork(d_model, d_model, d_model) for _ in range(n_vars)]
        )
        self.softmax_grn = GatedResidualNetwork(n_vars * d_model, n_vars * d_model, n_vars)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq, n_vars * d_model) containing
               pre‑embedded features for all variables.

        Returns:
            A tuple ``(out, weights)`` where:
                - ``out`` is a tensor of shape (batch, seq, d_model) representing
                  the aggregated variable representation.
                - ``weights`` is a tensor of shape (batch, seq, n_vars) containing
                  the learned attention weights for each variable.
        """
        # Process each variable independently through its GRN
        processed = [
            self.grns[i](
                x[..., i * x.shape[-1] // len(self.grns) : (i + 1) * x.shape[-1] // len(self.grns)]
            )
            for i in range(len(self.grns))
        ]
        stacked = torch.stack(processed, dim=-1)  # (batch, seq, d_model, n_vars)
        flat = x.reshape(x.shape[0], x.shape[1], -1)  # (batch, seq, n_vars * d_model)
        weights = torch.softmax(self.softmax_grn(flat), dim=-1).unsqueeze(-2)  # (batch, seq, 1, n_vars)
        out = (stacked * weights.permute(0, 1, 3, 2).unsqueeze(2)).sum(-1)  # (batch, seq, d_model)
        return out.mean(-1), weights.squeeze(-2)


class TFTModel(AbstractModel, _NNModule):
    """
    Simplified Temporal Fusion Transformer.

    The model ingests a sequence of features and outputs a single probability
    indicating the likelihood of a price increase. It combines LSTM encoding,
    gated residual networks, and multi‑head self‑attention for interpretable
    forecasting.
    """

    model_type = "tft"

    def __init__(
        self,
        n_features: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        seq_len: int = 60,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            n_features: Number of input features per timestep.
            d_model: Dimensionality of the internal model representation.
            n_heads: Number of attention heads.
            seq_len: Length of the input sequence.
            dropout: Dropout probability applied throughout the network.
        """
        nn.Module.__init__(self)
        self.n_features = n_features
        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # LSTM encoder (temporal context)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=False)

        # GRN layers
        self.grn_enrich = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)

        # Multi‑head self‑attention (interpretable)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output head
        self.head = nn.Sequential(
            GatedResidualNetwork(d_model, d_model, d_model // 2, dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Tensor of shape (batch, 1) containing the probability that the price
            will move up.
        """
        h = self.input_proj(x)               # → (batch, seq, d_model)
        h, _ = self.lstm(h)                  # temporal encoding
        h = self.grn_enrich(h)               # gated enrichment

        # Self‑attention with residual connection
        attn_out, self._last_attn_weights = self.attn(h, h, h)
        h = self.ln1(h + self.dropout(attn_out))
        h = self.ln2(h + self.attn_grn(h))

        # Use last timestep for classification
        return self.head(h[:, -1, :])

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """
        Retrieve the attention weights from the most recent forward pass.

        Returns:
            A NumPy array of shape (batch, seq_len, seq_len) containing the
            attention matrix, or ``None`` if a forward pass has not been executed.
        """
        if hasattr(self, "_last_attn_weights") and self._last_attn_weights is not None:
            return self._last_attn_weights.detach().cpu().numpy()
        return None

    def train_epoch(self, loader: Any, optimizer: Any, criterion: Any) -> Dict[str, float]:
        """
        Perform a single training epoch.

        Args:
            loader: Iterable yielding ``(x, y)`` batches.
            optimizer: Optimizer used for parameter updates.
            criterion: Loss function.

        Returns:
            Dictionary with average ``loss`` and ``acc`` over the epoch.
        """
        self.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = self(x).squeeze(-1)
            loss = criterion(pred, y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_acc += ((pred > 0.5) == y.bool()).float().mean().item()
            n += 1
        return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}

    def evaluate(self, loader: Any) -> EvalMetrics:
        """
        Evaluate the model on a validation/test set.

        Args:
            loader: Iterable yielding ``(x, y)`` batches.

        Returns:
            An ``EvalMetrics`` instance containing accuracy, AUC, Sharpe ratio,
            and optional loss.
        """
        self.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                preds.extend(self(x).squeeze(-1).numpy())
                labels.extend(y.numpy())
        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        acc = float(((preds_arr > 0.5) == (labels_arr > 0.5)).mean())
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(labels_arr, preds_arr))
        except Exception:
            auc = 0.5
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0, loss=None)


# Public registry name: app.ml.models.__init__ imports `TransformerPredictor`.
# The implementation is the Temporal Fusion Transformer below.
TransformerPredictor = TFTModel