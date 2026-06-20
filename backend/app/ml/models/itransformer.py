"""
iTransformer (Liu et al., ICLR 2024) — pure PyTorch implementation.

Key innovation: invert the attention axis — treat each feature/variate as a
token (not each time step). The transformer learns cross-feature dependencies
by attending over the variate dimension, where each token summarises the full
time series of one feature via a Linear projection.

Architecture:
  Input: (batch, seq_len, n_features)
  Step 1 — Variate embedding:
    For each variate i, embed its time series [x_{1,i}...x_{T,i}]
    → d_model via Linear(seq_len, d_model)                        → (B, F, D)
  Step 2 — Inverted encoder (N layers, Pre-LN):
    LayerNorm → MultiHeadAttention (Q/K/V over F variates) → Add
    LayerNorm → FFN (d_ff, GELU) → Add
  Step 3 — Head:
    Mean pool over variates → LayerNorm → Linear(d_model, 1) → squeeze → (B,)
    (sigmoid applied in predict_proba / train; raw logits for BCEWithLogitsLoss)

Exports:
  iTransformer   — model class
  train(...)     — async training entry point matching train_lstm.py API
"""
from __future__ import annotations

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    DataLoader = None    # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

# Real nn.Module base when torch is present; ``object`` fallback so the class still
# imports (as an inert placeholder) without torch. Instantiation still requires torch.
_NNModule = nn.Module if _TORCH_AVAILABLE else object

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from app.ml.models.base_model import AbstractModel, EvalMetrics


# ---------------------------------------------------------------------------
# Inverted Encoder Layer
# ---------------------------------------------------------------------------

class InvertedEncoderLayer(_NNModule):
    """
    Single Pre-LN transformer layer where attention is computed over the
    variate (feature) dimension rather than the time dimension.

    Input / output: (batch, n_variates, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_variates, d_model)
        Returns:
            (batch, n_variates, d_model)
        """
        # Self-attention over variate tokens (Pre-LN)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop1(attn_out)

        # Feed-forward (Pre-LN)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# iTransformer
# ---------------------------------------------------------------------------

class iTransformer(AbstractModel, _NNModule):
    """
    iTransformer: inverted-attention transformer for multivariate time series.

    Each variate (feature) is embedded from its full time series into a single
    d_model token; transformer layers then learn cross-variate dependencies.
    """
    model_type = "itransformer"

    def __init__(
        self,
        n_features: int = 27,
        seq_len: int = 60,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_p = dropout

        # Step 1 — Variate embedding: Linear(seq_len → d_model) shared across variates
        self.variate_embed = nn.Linear(seq_len, d_model)

        # Optional learnable variate-position embedding
        self.variate_pos = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.variate_pos, std=0.02)

        self.embed_drop = nn.Dropout(dropout)

        # Step 2 — Inverted encoder layers
        self.encoder = nn.ModuleList([
            InvertedEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Step 3 — Classification head
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch,) — raw logits (apply sigmoid for probabilities)
        """
        B, T, F = x.shape

        # Transpose: (B, F, T) so each variate has its time series as a vector
        x = x.permute(0, 2, 1)  # (B, n_features, seq_len)

        # Handle seq_len mismatch gracefully (pad / truncate)
        if T < self.seq_len:
            pad = x.new_zeros(B, F, self.seq_len - T)
            x = torch.cat([x, pad], dim=-1)
        elif T > self.seq_len:
            x = x[:, :, :self.seq_len]

        # Variate embedding: each row (seq_len,) → d_model
        x = self.variate_embed(x)           # (B, F, d_model)
        x = x + self.variate_pos[:, :F, :]  # add positional embedding
        x = self.embed_drop(x)

        # Inverted encoder: attention over variate dimension
        for layer in self.encoder:
            x = layer(x)                    # (B, F, d_model)

        # Head: mean pool over variates
        x = x.mean(dim=1)                   # (B, d_model)
        x = self.head_norm(x)
        logits = self.head(x).squeeze(-1)   # (B,)
        return logits

    # ------------------------------------------------------------------
    # AbstractModel interface
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader, optimizer, criterion) -> dict:
        """Train for one epoch. Returns dict with 'loss' and 'accuracy'."""
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

    def evaluate(self, loader: DataLoader) -> EvalMetrics:
        """Evaluate model on a DataLoader. Returns EvalMetrics."""
        self.eval()
        all_logits, all_labels = [], []
        total_loss, total = 0.0, 0
        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for X, y in loader:
                logits = self.forward(X)
                loss = criterion(logits, y.float())
                total_loss += loss.item() * len(y)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
                total += len(y)

        logits_cat = torch.cat(all_logits).numpy()
        labels_cat = torch.cat(all_labels).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits_cat))
        preds = (probs > 0.5).astype(int)
        acc = float((preds == labels_cat).mean())

        if _HAS_SKLEARN:
            try:
                auc = float(roc_auc_score(labels_cat, probs))
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5

        return EvalMetrics(
            accuracy=acc,
            auc=auc,
            sharpe=0.0,
            loss=total_loss / max(total, 1),
        )


# ---------------------------------------------------------------------------
# Training entry point (matches train_lstm.py API)
# ---------------------------------------------------------------------------

async def train(
    ohlcv_df,
    experiment_name: str = "itransformer_default",
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 3,
    d_ff: int = 512,
    dropout: float = 0.1,
    seq_len: int = 60,
    max_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
) -> dict:
    """
    Train an iTransformer model on an OHLCV DataFrame.

    Returns a results dict with loss, accuracy, and artifact_path.
    The walk-forward / temporal split is shuffle=False to prevent lookahead.
    """
    from app.ml.features.engineer import engineer_features, create_sequences, add_labels
    from app.ml.training.trainer import train_with_lightning, ARTIFACTS_DIR

    df = engineer_features(ohlcv_df)
    df = add_labels(df, threshold=0.002)
    X, y = create_sequences(df, seq_len=seq_len)

    n_features = X.shape[2]
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = iTransformer(
        n_features=n_features,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    )

    results = train_with_lightning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name,
        max_epochs=max_epochs,
        lr=lr,
    )

    save_path = ARTIFACTS_DIR / experiment_name / "final_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "experiment": experiment_name,
    }, str(save_path))

    results["artifact_path"] = str(save_path)
    return results
