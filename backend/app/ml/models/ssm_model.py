"""
Simplified State Space Model (SSM) in pure PyTorch.

Inspired by S4/Mamba without requiring CUDA compilation. Uses a diagonal
structured state matrix (DSS variant) that is numerically stable and fast
on CPU. Same interface as LSTMPredictor.

Architecture:
  Input:  (batch, seq_len, n_features)
  → Embedding linear
  → N × SSM layers (each: selective state space + feed-forward)
  → Mean pooling over sequence
  → Linear(d_model → 1) → Sigmoid
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]

import numpy as np

from app.ml.models.base_model import AbstractModel, EvalMetrics


class _SSMLayer(nn.Module):
    """
    Single SSM layer: diagonal structured state space + feed-forward residual.

    State equation (discretized via ZOH):
        h_t = exp(A * dt) * h_{t-1} + B * x_t
        y_t = C * h_t + D * x_t
    where A is a learned diagonal matrix (log-parameterised for stability).
    """

    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal SSM parameters (log A for stability, init to small negatives)
        self.log_A = nn.Parameter(torch.randn(d_model, d_state) * 0.5 - 2.0)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_model) * 0.01)

        # Learned step size (dt = softplus(dt_proj))
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        # Feed-forward residual
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential scan over time steps. x: (B, T, D)."""
        B, T, D = x.shape
        N = self.d_state

        # Compute step size: (B, T, D) → (B, T, D) positive via softplus
        dt = F.softplus(self.dt_proj(x))                # (B, T, D)

        # Discretize A: exp(A * dt), A is (D, N), dt is (B, T, D)
        A = -torch.exp(self.log_A)                       # (D, N), negative ensures stability
        # A_bar: (B, T, D, N)
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # B_bar: (B, T, D, N) = dt * B (simplified ZOH)
        B_bar = dt.unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0)  # (B, T, D, N)

        # Sequential scan
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            # h: (B, D, N), A_bar[t]: (B, D, N), B_bar[t]: (B, D, N)
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # y_t = C * h summed over state + D * x_t
            y_t = (self.C.unsqueeze(0) * h).sum(-1) + self.D * x[:, t]  # (B, D)
            outs.append(y_t.unsqueeze(1))

        return torch.cat(outs, dim=1)  # (B, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        # SSM sub-layer with residual
        x = x + self.drop(self._ssm_scan(self.norm1(x)))
        # Feed-forward sub-layer with residual
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class SSMPredictor(AbstractModel, nn.Module):
    """
    SSM-based predictor for financial time series direction prediction.
    Same interface as LSTMPredictor — drop-in replacement in the ensemble.
    """
    model_type = "ssm"

    def __init__(
        self,
        n_features: int = 27,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.n_features = n_features
        self.d_model = d_model

        self.embedding = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList([
            _SSMLayer(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) → (batch, 1)"""
        x = self.embedding(x)            # (B, T, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)               # (B, d_model) — mean pool over time
        return self.head(x)              # (B, 1)

    # ── AbstractModel interface ───────────────────────────────────────────────

    def train_epoch(self, loader, optimizer, criterion) -> dict:
        if not _TORCH_AVAILABLE:
            return {"loss": 0.0}
        self.train()
        total_loss, n = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = self(xb).squeeze(-1)
            loss = criterion(pred, yb.float())
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        return {"loss": total_loss / max(n, 1)}

    def evaluate(self, loader) -> EvalMetrics:
        if not _TORCH_AVAILABLE:
            return EvalMetrics(accuracy=0.5, auc=0.5, sharpe=0.0)
        from sklearn.metrics import roc_auc_score
        self.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                p = self(xb).squeeze(-1).cpu().numpy()
                preds.extend(p.tolist())
                labels.extend(yb.cpu().numpy().tolist())
        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        acc = float(((preds_arr > 0.5) == labels_arr).mean())
        try:
            auc = float(roc_auc_score(labels_arr, preds_arr))
        except Exception:
            auc = 0.5
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    def save(self, path: str, metadata: dict | None = None) -> None:
        if not _TORCH_AVAILABLE:
            return
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "n_features": self.n_features,
                "d_model": self.d_model,
                "n_layers": len(self.layers),
            },
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str) -> SSMPredictor:
        if not _TORCH_AVAILABLE:
            return cls()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        model = cls(
            n_features=cfg.get("n_features", 27),
            d_model=cfg.get("d_model", 64),
            n_layers=cfg.get("n_layers", 4),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
