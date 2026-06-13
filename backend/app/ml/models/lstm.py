"""
Bidirectional LSTM with self-attention for time series direction prediction.

Architecture:
  Input: (batch, seq_len, n_features)
  → Bidirectional LSTM (hidden=128, layers=2, dropout=0.3)
  → Self-attention layer (learn which time steps matter)
  → LayerNorm → Linear(256→64) → GELU → Dropout(0.3)
  → Linear(64→1) → Sigmoid  [binary classification]
"""
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
import numpy as np
from sklearn.metrics import roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x)                  # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)       # (batch, seq, 1)
        return (weights * x).sum(dim=1)             # (batch, hidden)


class LSTMPredictor(AbstractModel, nn.Module):
    model_type = "lstm"

    def __init__(
        self,
        n_features: int = 27,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        nn.Module.__init__(self)
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        dirs = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.attention = SelfAttention(hidden_size * dirs)
        self.norm = nn.LayerNorm(hidden_size * dirs)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * dirs, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)               # (batch, seq, hidden*dirs)
        ctx = self.attention(out)            # (batch, hidden*dirs)
        ctx = self.norm(ctx)
        return self.head(ctx).squeeze(-1)   # (batch,) logits

    def train_epoch(self, loader, optimizer, criterion) -> dict:
        self.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            optimizer.zero_grad()
            logits = self.forward(X)
            loss = criterion(logits, y.float())
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += len(y)
        return {"loss": total_loss / total, "accuracy": correct / total}

    def evaluate(self, loader) -> EvalMetrics:
        self.eval()
        all_logits, all_labels = [], []
        total_loss, total = 0.0, 0
        criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            for X, y in loader:
                logits = self.forward(X)
                loss = criterion(logits, y.float())
                total_loss += loss.item() * len(y)
                all_logits.append(logits)
                all_labels.append(y)
                total += len(y)
        logits_cat = torch.cat(all_logits).numpy()
        labels_cat = torch.cat(all_labels).numpy()
        probs = 1 / (1 + np.exp(-logits_cat))
        preds = (probs > 0.5).astype(int)
        acc = (preds == labels_cat).mean()
        try:
            auc = float(roc_auc_score(labels_cat, probs))
        except ValueError:
            auc = 0.5
        return EvalMetrics(accuracy=float(acc), auc=auc, sharpe=0.0, loss=total_loss / total)
