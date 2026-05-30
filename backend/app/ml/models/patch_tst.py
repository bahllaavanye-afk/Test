"""
PatchTST (Nie et al., ICLR 2023) — pure PyTorch implementation.

Channel-independent mode: each feature dimension is processed through its own
Transformer encoder; predictions are averaged across channels.

Architecture per channel:
  Input: (batch, seq_len, n_features) → split to n_features channels
  Each channel: (batch, seq_len) → patches → Linear(patch_len, d_model)
               → + learnable pos embedding
               → TransformerEncoder (n_layers, n_heads)
               → mean pool over patches → LN → Linear(d_model, 1)
  Output: mean over n_features channels → (batch,)

Exports:
  PatchTST              — model class
  PatchEncoder          — reusable patch-embedding + transformer block
  train(...)            — async training entry point matching train_lstm.py API
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

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

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from app.ml.models.base_model import AbstractModel, EvalMetrics
from app.ml.features.engineer import engineer_features, create_sequences, add_labels
from app.ml.training.trainer import ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# PatchEncoder — reusable patch-embedding + TransformerEncoder block
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """
    Encodes a 1-D sequence into a fixed-size embedding via patching.

    Input:  (batch, seq_len)          — single channel
    Output: (batch, d_model)          — pooled representation

    Steps:
      1. Unfold (seq_len) into non-overlapping patches of size patch_len
         → shape (batch, n_patches, patch_len)
      2. Project each patch: Linear(patch_len, d_model)
      3. Add learnable positional embedding (n_patches, d_model)
      4. TransformerEncoder (n_layers, n_heads, d_model)
      5. Mean-pool over n_patches → (batch, d_model)
    """

    def __init__(
        self,
        seq_len: int = 64,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model

        # Number of patches (pad seq_len to be divisible by patch_len)
        self.n_patches = math.ceil(seq_len / patch_len)
        self.padded_len = self.n_patches * patch_len

        # Patch projection
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Learnable positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len)
        Returns:
            (batch, d_model)
        """
        B, T = x.shape
        # Pad to padded_len if needed
        if T < self.padded_len:
            pad = x.new_zeros(B, self.padded_len - T)
            x = torch.cat([x, pad], dim=1)
        elif T > self.padded_len:
            x = x[:, :self.padded_len]

        # Reshape to patches: (B, n_patches, patch_len)
        x = x.view(B, self.n_patches, self.patch_len)

        # Project patches
        x = self.patch_proj(x)              # (B, n_patches, d_model)
        x = x + self.pos_emb               # broadcast positional embedding

        # Transformer
        x = self.transformer(x)             # (B, n_patches, d_model)

        # Mean pool over patches
        x = x.mean(dim=1)                  # (B, d_model)
        return self.norm(x)


# ---------------------------------------------------------------------------
# PatchTST
# ---------------------------------------------------------------------------

class PatchTST(AbstractModel, nn.Module):
    """
    PatchTST: channel-independent patch-based time series transformer.

    In channel-independent mode each feature (channel) is processed
    independently through a shared PatchEncoder; their logits are averaged.
    """
    model_type = "patch_tst"

    def __init__(
        self,
        n_features: int = 27,
        seq_len: int = 64,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
        channel_independent: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.n_features = n_features
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.channel_independent = channel_independent

        if channel_independent:
            # One shared PatchEncoder for all channels
            self.encoder = PatchEncoder(
                seq_len=seq_len,
                patch_len=patch_len,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            )
            self.head = nn.Linear(d_model, 1)
        else:
            # Single encoder treating all features as one multivariate sequence
            self.encoder = PatchEncoder(
                seq_len=seq_len,
                patch_len=patch_len,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            )
            # Flatten input across features
            self._flat_proj = nn.Linear(n_features, 1)
            self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch,) — logits
        """
        if self.channel_independent:
            B, T, C = x.shape
            # Process each channel independently: (B*C, T)
            x_flat = x.permute(0, 2, 1).reshape(B * C, T)   # (B*C, T)
            h = self.encoder(x_flat)                          # (B*C, d_model)
            logits = self.head(h)                             # (B*C, 1)
            logits = logits.view(B, C).mean(dim=1)            # (B,)
        else:
            B, T, C = x.shape
            # Project features to single channel per time step
            x_1d = self._flat_proj(x).squeeze(-1)            # (B, T)
            h = self.encoder(x_1d)                            # (B, d_model)
            logits = self.head(h).squeeze(-1)                 # (B,)
        return logits

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
    experiment_name: str = "patch_tst_default",
    patch_len: int = 16,
    d_model: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
    dropout: float = 0.2,
    seq_len: int = 64,
    max_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 5e-4,
) -> dict:
    """
    Train a PatchTST model on an OHLCV DataFrame.
    Returns a results dict with loss, accuracy, and artifact_path.
    """
    from app.ml.training.trainer import train_with_lightning

    # Build features and sequences
    df = engineer_features(ohlcv_df)
    df = add_labels(df, threshold=0.002)
    X, y = create_sequences(df, seq_len=seq_len)

    n_features = X.shape[2]
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    test_ds = TensorDataset(X[n_train + n_val:], y[n_train + n_val:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = PatchTST(
        n_features=n_features,
        seq_len=seq_len,
        patch_len=patch_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        channel_independent=True,
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
        "patch_len": patch_len,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "experiment": experiment_name,
    }, str(save_path))

    results["artifact_path"] = str(save_path)
    return results
