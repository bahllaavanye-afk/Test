"""
Temporal Fusion Transformer (Lim et al., 2021).
State-of-the-art for multi-horizon time series forecasting.
Attention mechanism provides interpretable feature importance per timestep.
"""
from __future__ import annotations
import logging
from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

_TORCH_AVAILABLE = torch is not None and nn is not None

# Configure module logger
_logger = logging.getLogger(__name__)

# Use the real nn.Module base when torch is present; fall back to ``object`` so these
# classes still *import* (as inert placeholders) in torch-free environments. The model
# registry can then expose them without crashing; instantiating them still needs torch.
_NNModule = nn.Module if _TORCH_AVAILABLE else object

import numpy as np
from app.ml.models.base_model import AbstractModel, EvalMetrics


class GatedLinearUnit(_NNModule):
    def __init__(self, d: int):
        super().__init__()
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Torch is required to instantiate GatedLinearUnit") from _IMPORT_ERROR
        self.fc = nn.Linear(d, d * 2)

    def forward(self, x):
        try:
            h = self.fc(x)
            return h[..., :h.shape[-1] // 2] * torch.sigmoid(h[..., h.shape[-1] // 2 :])
        except Exception as exc:
            _logger.error("GatedLinearUnit forward failed", exc_info=exc)
            raise


class GatedResidualNetwork(_NNModule):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Torch is required to instantiate GatedResidualNetwork") from _IMPORT_ERROR
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = GatedLinearUnit(d_out)
        self.ln = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        try:
            h = torch.relu(self.fc1(x))
            h = self.dropout(h)
            h = self.fc2(h)
            h = self.gate(h) + self.skip(x)
            return self.ln(h)
        except Exception as exc:
            _logger.error("GatedResidualNetwork forward failed", exc_info=exc)
            raise


class VariableSelectionNetwork(_NNModule):
    """Softmax-weighted GRN per variable — tells us which features matter."""
    def __init__(self, n_vars: int, d_model: int):
        super().__init__()
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Torch is required to instantiate VariableSelectionNetwork") from _IMPORT_ERROR
        self.grns = nn.ModuleList(
            [GatedResidualNetwork(d_model, d_model, d_model) for _ in range(n_vars)]
        )
        self.softmax_grn = GatedResidualNetwork(
            n_vars * d_model, n_vars * d_model, n_vars
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            # x: (batch, seq, n_vars * d_model) — pre-embedded features
            processed = [
                self.grns[i](
                    x[
                        ..., i * x.shape[-1] // len(self.grns) : (i + 1) * x.shape[-1] // len(self.grns)
                    ]
                )
                for i in range(len(self.grns))
            ]
            stacked = torch.stack(processed, dim=-1)  # (batch, seq, d, n_vars)
            flat = x.reshape(x.shape[0], x.shape[1], -1)
            weights = torch.softmax(self.softmax_grn(flat), dim=-1).unsqueeze(-2)  # (batch, seq, 1, n_vars)
            out = (stacked * weights.permute(0, 1, 3, 2).unsqueeze(2)).sum(-1)
            return out.mean(-1), weights.squeeze(-2)
        except Exception as exc:
            _logger.error("VariableSelectionNetwork forward failed", exc_info=exc)
            raise


class TFTModel(AbstractModel, _NNModule):
    """
    Simplified Temporal Fusion Transformer.
    Input: (batch, seq_len, n_features)
    Output: (batch, 1) — probability of price up
    """
    model_type = "tft"

    def __init__(
        self,
        n_features: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        seq_len: int = 60,
        dropout: float = 0.1,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Torch is required to instantiate TFTModel") from _IMPORT_ERROR
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

        # Multi-head self-attention (interpretable)
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
        Forward pass through the TFT model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq, features).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 1) with probability predictions.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        try:
            # x: (batch, seq, features)
            h = self.input_proj(x)  # → (batch, seq, d_model)
            h, _ = self.lstm(h)  # temporal encoding
            h = self.grn_enrich(h)  # gated enrichment

            # Self-attention with residual
            attn_out, self._last_attn_weights = self.attn(h, h, h)
            h = self.ln1(h + self.dropout(attn_out))
            h = self.ln2(h + self.attn_grn(h))

            # Use last timestep for classification
            return self.head(h[:, -1, :])
        except Exception as exc:
            _logger.error("TFTModel forward pass failed", exc_info=exc)
            raise

    def get_attention_weights(self) -> np.ndarray | None:
        """Returns attention weights for interpretability (last forward pass)."""
        try:
            if hasattr(self, "_last_attn_weights") and self._last_attn_weights is not None:
                return self._last_attn_weights.detach().cpu().numpy()
            return None
        except Exception as exc:
            _logger.error("Failed to retrieve attention weights", exc_info=exc)
            return None

    def train_epoch(self, loader, optimizer, criterion) -> dict:
        """
        Runs a single training epoch.

        Parameters
        ----------
        loader : iterable
            Data loader yielding (x, y) batches.
        optimizer : torch.optim.Optimizer
            Optimizer instance.
        criterion : callable
            Loss function.

        Returns
        -------
        dict
            Dictionary with average loss and accuracy.
        """
        self.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch_idx, (x, y) in enumerate(loader, start=1):
            try:
                optimizer.zero_grad()
                pred = self(x).squeeze(-1)
                loss = criterion(pred, y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                total_acc += ((pred > 0.5) == y.bool()).float().mean().item()
                n += 1
            except Exception as exc:
                _logger.error(
                    f"Training batch {batch_idx} failed", exc_info=exc, extra={"batch": batch_idx}
                )
                raise
        return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}

    def evaluate(self, loader) -> EvalMetrics:
        """
        Evaluates the model on a validation/test loader.

        Parameters
        ----------
        loader : iterable
            Data loader yielding (x, y) batches.

        Returns
        -------
        EvalMetrics
            Evaluation metrics containing accuracy, AUC, etc.
        """
        self.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loader, start=1):
                try:
                    preds.extend(self(x).squeeze(-1).cpu().numpy())
                    labels.extend(y.cpu().numpy())
                except Exception as exc:
                    _logger.error(
                        f"Evaluation batch {batch_idx} failed", exc_info=exc, extra={"batch": batch_idx}
                    )
                    raise
        preds = np.array(preds)
        labels = np.array(labels)
        acc = float(((preds > 0.5) == (labels > 0.5)).mean())
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(labels, preds))
        except Exception as exc:
            _logger.warning("AUC calculation failed; defaulting to 0.5", exc_info=exc)
            auc = 0.5
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0, loss=None)


# Public registry name: app.ml.models.__init__ imports `TransformerPredictor`.
# The implementation is the Temporal Fusion Transformer below.
TransformerPredictor = TFTModel