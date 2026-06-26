"""
Temporal Fusion Transformer (Lim et al., 2021).
State-of-the-art for multi-horizon time series forecasting.
Attention mechanism provides interpretable feature importance per timestep.
"""
from __future__ import annotations

import logging
import time
from typing import Any

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
import numpy as np

from app.ml.models.base_model import AbstractModel, EvalMetrics

_logger = logging.getLogger(__name__)


def _log_structured(event: str, **metrics: Any) -> None:
    """Emit a structured log record at INFO level."""
    if _logger.isEnabledFor(logging.INFO):
        _logger.info(event, extra=metrics)


class GatedLinearUnit(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, d * 2)

    def forward(self, x):
        h = self.fc(x)
        return h[..., :h.shape[-1] // 2] * torch.sigmoid(h[..., h.shape[-1] // 2:])


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = GatedLinearUnit(d_out)
        self.ln = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.gate(h) + self.skip(x)
        return self.ln(h)


class VariableSelectionNetwork(nn.Module):
    """Softmax-weighted GRN per variable — tells us which features matter."""
    def __init__(self, n_vars: int, d_model: int):
        super().__init__()
        self.grns = nn.ModuleList([GatedResidualNetwork(d_model, d_model, d_model) for _ in range(n_vars)])
        self.softmax_grn = GatedResidualNetwork(n_vars * d_model, n_vars * d_model, n_vars)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq, n_vars * d_model) — pre-embedded features
        processed = [
            self.grns[i](
                x[..., i * x.shape[-1] // len(self.grns) : (i + 1) * x.shape[-1] // len(self.grns)]
            )
            for i in range(len(self.grns))
        ]
        stacked = torch.stack(processed, dim=-1)   # (batch, seq, d, n_vars)
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        weights = torch.softmax(self.softmax_grn(flat), dim=-1).unsqueeze(-2)  # (batch, seq, 1, n_vars)
        out = (stacked * weights.permute(0, 1, 3, 2).unsqueeze(2)).sum(-1)
        return out.mean(-1), weights.squeeze(-2)


class TFTModel(AbstractModel, nn.Module):
    """
    Simplified Temporal Fusion Transformer.
    Input: (batch, seq_len, n_features)
    Output: (batch, 1) — probability of price up
    """
    model_type = "tft"

    def __init__(self, n_features: int = 20, d_model: int = 64, n_heads: int = 4,
                 seq_len: int = 60, dropout: float = 0.1):
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
        """Run a forward pass and emit structured metrics."""
        start_ts = time.time()
        # x: (batch, seq, features)
        h = self.input_proj(x)                      # → (batch, seq, d_model)
        h, _ = self.lstm(h)                          # temporal encoding
        h = self.grn_enrich(h)                       # gated enrichment

        # Self-attention with residual
        attn_out, self._last_attn_weights = self.attn(h, h, h)
        h = self.ln1(h + self.dropout(attn_out))
        h = self.ln2(h + self.attn_grn(h))

        out = self.head(h[:, -1, :])
        exec_time_ms = (time.time() - start_ts) * 1000
        signal_count = int(x.shape[0] * x.shape[1])  # batch * seq_len
        _log_structured(
            "tft_forward",
            signal_count=signal_count,
            execution_time_ms=exec_time_ms,
            pnl=None,
        )
        return out

    def get_attention_weights(self) -> np.ndarray | None:
        """Returns attention weights for interpretability (last forward pass)."""
        if hasattr(self, "_last_attn_weights") and self._last_attn_weights is not None:
            return self._last_attn_weights.detach().cpu().numpy()
        return None

    def train_epoch(self, loader, optimizer, criterion) -> dict:
        """Train for one epoch and log aggregate metrics."""
        epoch_start = time.time()
        self.train()
        total_loss, total_acc, total_signals, n = 0.0, 0.0, 0, 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = self(x).squeeze(-1)
            loss = criterion(pred, y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            batch_signals = int(x.shape[0] * x.shape[1])
            total_signals += batch_signals
            total_loss += loss.item()
            total_acc += ((pred > 0.5) == y.bool()).float().mean().item()
            n += 1
        epoch_time_ms = (time.time() - epoch_start) * 1000
        avg_loss = total_loss / max(n, 1)
        avg_acc = total_acc / max(n, 1)
        _log_structured(
            "tft_train_epoch",
            signal_count=total_signals,
            execution_time_ms=epoch_time_ms,
            avg_loss=avg_loss,
            avg_accuracy=avg_acc,
            pnl=None,
        )
        return {"loss": avg_loss, "acc": avg_acc}

    def evaluate(self, loader) -> EvalMetrics:
        """Evaluate on a validation set and log key performance indicators."""
        eval_start = time.time()
        self.eval()
        preds, labels = [], []
        total_signals = 0
        with torch.no_grad():
            for x, y in loader:
                batch_signals = int(x.shape[0] * x.shape[1])
                total_signals += batch_signals
                preds.extend(self(x).squeeze(-1).numpy())
                labels.extend(y.numpy())
        preds = np.array(preds)
        labels = np.array(labels)
        acc = float(((preds > 0.5) == (labels > 0.5)).mean())
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(labels, preds))
        except Exception:
            auc = 0.5
        exec_time_ms = (time.time() - eval_start) * 1000
        _log_structured(
            "tft_evaluate",
            signal_count=total_signals,
            execution_time_ms=exec_time_ms,
            accuracy=acc,
            auc=auc,
            pnl=None,
        )
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0, loss=None)


# Backward-compatible alias — registry imports this name.
TransformerPredictor = TFTModel