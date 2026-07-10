"""
PatchTST (Nie et al., ICLR 2023) — pure PyTorch implementation.

Channel‑independent mode: each feature dimension is processed through its own
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
  PatchEncoder          — reusable patch‑embedding + transformer block
  train(...)            — async training entry point matching train_lstm.py API
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

from app.ml.models.base_model import AbstractModel, EvalMetrics
from app.ml.features.engineer import engineer_features, create_sequences, add_labels
from app.ml.training.trainer import ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# PatchEncoder — reusable patch‑embedding + TransformerEncoder block
# ---------------------------------------------------------------------------


class PatchEncoder(nn.Module):
    """
    Encode a 1‑D sequence into a fixed‑size embedding via patching.

    The encoder works on a single channel (shape ``(batch, seq_len)``) and
    returns a pooled representation of size ``d_model`` (shape ``(batch,
    d_model)``).

    Steps
    -----
    1. Unfold the input sequence into non‑overlapping patches of length
       ``patch_len`` → shape ``(batch, n_patches, patch_len)``.
    2. Project each patch with a linear layer ``Linear(patch_len, d_model)``.
    3. Add a learnable positional embedding of shape ``(1, n_patches, d_model)``.
    4. Apply a ``TransformerEncoder`` consisting of ``n_layers`` layers and
       ``n_heads`` attention heads.
    5. Mean‑pool over the patch dimension → ``(batch, d_model)``.
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
        """
        Initialise a :class:`PatchEncoder`.

        Parameters
        ----------
        seq_len: int
            Length of the input time‑series.
        patch_len: int
            Length of each patch; the sequence is padded to be divisible by this.
        d_model: int
            Dimensionality of the transformer model.
        n_heads: int
            Number of attention heads.
        n_layers: int
            Number of transformer encoder layers.
        dropout: float
            Dropout probability applied inside the transformer.
        """
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
            norm_first=True,  # Pre‑LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Pooled embedding of shape ``(batch, d_model)``.
        """
        B, T = x.shape
        # Pad or truncate to ``padded_len`` if necessary
        if T < self.padded_len:
            pad = x.new_zeros(B, self.padded_len - T)
            x = torch.cat([x, pad], dim=1)
        elif T > self.padded_len:
            x = x[:, : self.padded_len]

        # Reshape to patches: (B, n_patches, patch_len)
        x = x.view(B, self.n_patches, self.patch_len)

        # Linear projection of each patch
        x = self.patch_proj(x)            # (B, n_patches, d_model)
        x = x + self.pos_emb              # broadcast positional embedding

        # Transformer encoder
        x = self.transformer(x)           # (B, n_patches, d_model)

        # Mean‑pool over patches and final LayerNorm
        x = x.mean(dim=1)                 # (B, d_model)
        return self.norm(x)


# ---------------------------------------------------------------------------
# PatchTST
# ---------------------------------------------------------------------------


class PatchTST(AbstractModel, nn.Module):
    """
    PatchTST: channel‑independent patch‑based time‑series transformer.

    In channel‑independent mode each feature (channel) is processed
    independently through a shared :class:`PatchEncoder`; their logits are
    averaged to produce the final prediction.
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
        """
        Initialise the PatchTST model.

        Parameters
        ----------
        n_features: int
            Number of input features (channels).
        seq_len: int
            Length of the input sequence.
        patch_len: int
            Length of each patch.
        d_model: int
            Transformer model dimension.
        n_heads: int
            Number of attention heads.
        n_layers: int
            Number of transformer encoder layers.
        dropout: float
            Dropout probability.
        channel_independent: bool
            If ``True``, each channel is processed independently with a shared
            encoder; otherwise the whole multivariate series is treated as a
            single sequence.
        """
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
            # Flatten input across features before encoding
            self._flat_proj = nn.Linear(n_features, 1)
            self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch,)``.
        """
        if self.channel_independent:
            B, T, C = x.shape
            # Process each channel independently: reshape to (B*C, T)
            x_flat = x.permute(0, 2, 1).reshape(B * C, T)  # (B*C, T)
            h = self.encoder(x_flat)                       # (B*C, d_model)
            logits = self.head(h)                           # (B*C, 1)
            logits = logits.view(B, C).mean(dim=1)          # (B,)
        else:
            B, T, C = x.shape
            # Project features to a single channel per time step
            x_1d = self._flat_proj(x).squeeze(-1)           # (B, T)
            h = self.encoder(x_1d)                          # (B, d_model)
            logits = self.head(h).squeeze(-1)               # (B,)
        return logits

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Train the model for a single epoch.

        Parameters
        ----------
        loader: DataLoader
            Iterable over training batches ``(X, y)``.
        optimizer: torch.optim.Optimizer
            Optimiser used to update model parameters.
        criterion: nn.Module
            Loss function (e.g., ``nn.BCEWithLogitsLoss``).

        Returns
        -------
        dict
            Mapping containing the average ``loss`` and ``accuracy`` for the
            epoch.
        """
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in loader:
            optimizer.zero_grad()
            logits = self.forward(X)
            loss = criterion(logits, y.float())
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y).sum().item()
            total += len(y)

        avg_loss = total_loss / total if total else 0.0
        accuracy = correct / total if total else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a validation set.

        Parameters
        ----------
        loader: DataLoader
            Validation data loader.
        criterion: nn.Module
            Loss function.

        Returns
        -------
        dict
            Dictionary with ``loss`` and ``accuracy`` (and ``auc`` if scikit‑learn
            is available).
        """
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels: list[int] = []
        all_scores: list[float] = []

        with torch.no_grad():
            for X, y in loader:
                logits = self.forward(X)
                loss = criterion(logits, y.float())
                total_loss += loss.item() * len(y)

                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == y).sum().item()
                total += len(y)

                if _HAS_SKLEARN:
                    all_labels.extend(y.cpu().tolist())
                    all_scores.extend(torch.sigmoid(logits).cpu().tolist())

        avg_loss = total_loss / total if total else 0.0
        accuracy = correct / total if total else 0.0
        metrics: Dict[str, float] = {"loss": avg_loss, "accuracy": accuracy}
        if _HAS_SKLEARN and all_labels:
            metrics["auc"] = float(roc_auc_score(all_labels, all_scores))
        return metrics

    async def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 20,
        patience: int = 5,
    ) -> Tuple[Dict[str, Any], Path]:
        """
        Async training loop compatible with the existing ``train_lstm.py`` API.

        Parameters
        ----------
        train_loader: DataLoader
            Training data loader.
        val_loader: DataLoader
            Validation data loader.
        optimizer: torch.optim.Optimizer
            Optimiser.
        criterion: nn.Module
            Loss function.
        epochs: int, default 20
            Maximum number of epochs.
        patience: int, default 5
            Early‑stopping patience based on validation loss.

        Returns
        -------
        tuple
            ``(best_metrics, artifact_path)`` where ``best_metrics`` contains the
            validation metrics of the best epoch and ``artifact_path`` points to
            the saved model checkpoint.
        """
        best_val_loss = float("inf")
        best_metrics: Dict[str, Any] = {}
        patience_counter = 0
        artifact_path = ARTIFACTS_DIR / f"{self.model_type}_latest.pt"

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            val_metrics = self.evaluate(val_loader, criterion)

            # Simple early‑stopping based on validation loss
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_metrics = val_metrics
                torch.save(self.state_dict(), artifact_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            # Mimic the original script's logging format
            log_entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            }
            if "auc" in val_metrics:
                log_entry["val_auc"] = val_metrics["auc"]
            print(json.dumps(log_entry))

            # Allow other async tasks to run
            await asyncio.sleep(0)

        return best_metrics, artifact_path

# End of file