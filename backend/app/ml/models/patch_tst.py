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
  PatchTSTConfig        — pydantic schema for model hyper‑parameters
  train(...)            — async training entry point matching train_lstm.py API
"""
from __future__ import annotations

import math
from typing import Any

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

# Pydantic is part of the core dependencies for schema validation.
try:
    from pydantic import BaseModel, Field, validator
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False
    BaseModel = object  # type: ignore[misc,assignment]
    Field = lambda *a, **kw: None  # type: ignore[assignment]
    validator = lambda *a, **kw: (lambda f: f)  # type: ignore[assignment]

from app.ml.features.engineer import add_labels, create_sequences, engineer_features
from app.ml.models.base_model import AbstractModel, EvalMetrics
from app.ml.training.trainer import ARTIFACTS_DIR

# ---------------------------------------------------------------------------
# PatchTST configuration schema
# ---------------------------------------------------------------------------

class PatchTSTConfig(BaseModel):
    """
    Configuration schema for :class:`PatchTST`.

    Provides validation, documentation and examples for each hyper‑parameter.
    """

    n_features: int = Field(
        27,
        ge=1,
        description="Number of input features (channels).",
        example=27,
    )
    seq_len: int = Field(
        64,
        ge=1,
        description="Length of the input sequence per feature.",
        example=64,
    )
    patch_len: int = Field(
        16,
        ge=1,
        description="Length of each non‑overlapping patch.",
        example=16,
    )
    d_model: int = Field(
        128,
        ge=1,
        description="Dimensionality of the embedding space.",
        example=128,
    )
    n_heads: int = Field(
        4,
        ge=1,
        description="Number of attention heads in the Transformer encoder.",
        example=4,
    )
    n_layers: int = Field(
        3,
        ge=1,
        description="Number of Transformer encoder layers.",
        example=3,
    )
    dropout: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Dropout probability applied inside the encoder.",
        example=0.2,
    )
    channel_independent: bool = Field(
        True,
        description="Whether to process each channel independently.",
        example=True,
    )

    @validator("patch_len")
    def _patch_len_not_exceed_seq_len(cls, v: int, values: dict[str, Any]) -> int:
        seq_len = values.get("seq_len")
        if seq_len is not None and v > seq_len:
            raise ValueError("patch_len must be less than or equal to seq_len")
        return v

    @validator("n_heads")
    def _heads_divisible_by_d_model(cls, v: int, values: dict[str, Any]) -> int:
        d_model = values.get("d_model")
        if d_model is not None and d_model % v != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return v

    class Config:
        """Pydantic configuration."""

        anystr_strip_whitespace = True
        validate_assignment = True


# ---------------------------------------------------------------------------
# PatchEncoder — reusable patch-embedding + TransformerEncoder block
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """
    Encodes a 1‑D sequence into a fixed‑size embedding via patching.

    Input:  (batch, seq_len)          — single channel
    Output: (batch, d_model)          — pooled representation

    Steps:
      1. Unfold (seq_len) into non‑overlapping patches of size patch_len
         → shape (batch, n_patches, patch_len)
      2. Project each patch: Linear(patch_len, d_model)
      3. Add learnable positional embedding (n_patches, d_model)
      4. TransformerEncoder (n_layers, n_heads, d_model)
      5. Mean‑pool over n_patches → (batch, d_model)
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
            norm_first=True,   # Pre‑LN for stability
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
    PatchTST: channel‑independent patch‑based time series transformer.

    In channel‑independent mode each feature (channel) is processed
    independently through a shared :class:`PatchEncoder`; their logits are
    averaged.
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

        return {
            "loss": total_loss / total if total else 0.0,
            "accuracy": correct / total if total else 0.0,
        }

# Exported symbols
__all__ = [
    "PatchEncoder",
    "PatchTST",
    "PatchTSTConfig",
]