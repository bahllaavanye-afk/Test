"""
MultiScaleTransformer — three-stream cross-attention transformer.

Three data streams at different temporal resolutions are fused via
cross-attention, optionally conditioned on macro signals (VIX/yield/USD).

Architecture:
  x_base  → PatchEncoder(seq=60, patch=8)   → h_base  (B, D)
  x_mid   → PatchEncoder(seq=20, patch=4)   → h_mid   (B, D)  [optional]
  x_slow  → PatchEncoder(seq=10, patch=2)   → h_slow  (B, D)  [optional]

  CrossAttn: Q=h_base, KV=h_mid  → h12
  CrossAttn: Q=h12,   KV=h_slow  → h123

  macro_embed = Linear(n_macro, D)(macro) if macro else zeros(B, D)

  cat(h_base, h12, h123, macro_embed)  →  LN  →  Linear(4D→D)  →  GELU
    →  Dropout  →  Linear(D→1)  →  squeeze  →  (B,)

Fallback (x_mid=None, x_slow=None):
  h_base  →  Linear(D→1)  →  squeeze  →  (B,)

Exports:
  MultiScaleTransformer  — model class
  train(...)             — async training entry point
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

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
from app.ml.models.patch_tst import PatchEncoder


# ---------------------------------------------------------------------------
# Cross-Attention module
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Single-head cross-attention: Q from one stream, K/V from another.

    Both inputs are (batch, d_model) vectors; we add a sequence dimension of 1
    so the standard MultiheadAttention can be used directly.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, d_model)
            kv:    (batch, d_model)

        Returns:
            (batch, d_model) — cross‑attended query
        """
        q = query.unsqueeze(1)   # (B, 1, D)
        k = kv.unsqueeze(1)      # (B, 1, D)
        out, _ = self.attn(q, k, k)  # (B, 1, D)
        out = out.squeeze(1)         # (B, D)
        return self.norm(out + query)  # residual + norm


# ---------------------------------------------------------------------------
# MultiScaleTransformer
# ---------------------------------------------------------------------------

class MultiScaleTransformer(AbstractModel, nn.Module):
    """
    Three‑stream cross‑attention transformer for multi‑resolution trading signals.
    """
    model_type = "multiscale_transformer"

    def __init__(
        self,
        n_features_base: int = 27,
        n_features_mid: int = 27,
        n_features_slow: int = 27,
        n_macro: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        nn.Module.__init__(self)
        self.n_features_base = n_features_base
        self.n_features_mid = n_features_mid
        self.n_features_slow = n_features_slow
        self.n_macro = n_macro
        self.d_model = d_model

        # --- Stream encoders ---
        self.enc_base = PatchEncoder(
            seq_len=60,
            patch_len=8,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.enc_mid = PatchEncoder(
            seq_len=20,
            patch_len=4,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.enc_slow = PatchEncoder(
            seq_len=10,
            patch_len=2,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # --- Channel‑independent projections (for multi‑feature inputs) ---
        self.proj_base = nn.Linear(n_features_base, 1)
        self.proj_mid = nn.Linear(n_features_mid, 1)
        self.proj_slow = nn.Linear(n_features_slow, 1)

        # --- Cross‑attention layers ---
        self.cross_base_mid = CrossAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.cross_12_slow = CrossAttention(d_model, n_heads=n_heads, dropout=dropout)

        # --- Macro conditioning ---
        self.macro_proj = nn.Linear(n_macro, d_model)

        # --- Fusion head ---
        self.fusion_norm = nn.LayerNorm(4 * d_model)
        self.fusion_head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # --- Single‑stream fallback head ---
        self.fallback_head = nn.Linear(d_model, 1)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _encode_stream(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
        encoder: PatchEncoder,
    ) -> torch.Tensor:
        """
        Encode a (batch, seq_len, n_features) tensor using channel projection
        followed by the PatchEncoder.

        Returns:
            (batch, d_model)
        """
        # Project features to a single channel: (B, T, 1) → (B, T)
        x_1d = proj(x).squeeze(-1)   # (B, T)
        return encoder(x_1d)          # (B, d_model)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        x_base: torch.Tensor,
        x_mid: torch.Tensor | None = None,
        x_slow: torch.Tensor | None = None,
        macro: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_base: (batch, 60, n_features_base)
            x_mid:  (batch, 20, n_features_mid)   or None
            x_slow: (batch, 10, n_features_slow)  or None
            macro:  (batch, n_macro)               or None

        Returns:
            (batch,) — logits
        """
        B = x_base.shape[0]

        # Encode base stream
        h_base = self._encode_stream(x_base, self.proj_base, self.enc_base)  # (B, D)

        if x_mid is None and x_slow is None:
            # Single‑stream fallback
            return self.fallback_head(h_base).squeeze(-1)  # (B,)

        # Encode mid and slow streams (use zero tensors when missing)
        if x_mid is not None:
            h_mid = self._encode_stream(x_mid, self.proj_mid, self.enc_mid)    # (B, D)
        else:
            h_mid = torch.zeros(B, self.d_model, device=x_base.device, dtype=x_base.dtype)

        if x_slow is not None:
            h_slow = self._encode_stream(x_slow, self.proj_slow, self.enc_slow)  # (B, D)
        else:
            h_slow = torch.zeros(B, self.d_model, device=x_base.device, dtype=x_base.dtype)

        # Cross‑attention fusion
        h12 = self.cross_base_mid(h_base, h_mid)      # (B, D)
        h123 = self.cross_12_slow(h12, h_slow)        # (B, D)

        # Macro conditioning
        if macro is not None:
            macro_emb = self.macro_proj(macro.float())   # (B, D)
        else:
            macro_emb = torch.zeros(B, self.d_model, device=x_base.device, dtype=x_base.dtype)

        # Fuse all streams
        fused = torch.cat([h_base, h12, h123, macro_emb], dim=-1)  # (B, 4D)
        fused = self.fusion_norm(fused)
        out = self.fusion_head(fused).squeeze(-1)  # (B,)
        return out

    # -----------------------------------------------------------------------
    # Training / evaluation utilities
    # -----------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader, optimizer: Any, criterion: Any) -> Dict[str, float]:
        """
        Perform a single training epoch.

        Args:
            loader:    DataLoader yielding batches of (x_base, x_mid, x_slow, macro, y)
            optimizer: Optimizer instance.
            criterion: Loss function (e.g., torch.nn.BCEWithLogitsLoss).

        Returns:
            Dictionary containing the average training loss.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training.")
        self.train()
        total_loss = 0.0
        total_samples = 0

        for batch in loader:
            optimizer.zero_grad()
            if not isinstance(batch, (list, tuple)):
                raise ValueError("Batch must be a tuple/list.")
            x_base, x_mid, x_slow, macro, y = batch

            preds = self(x_base, x_mid, x_slow, macro)
            loss = criterion(preds, y.float())
            loss.backward()
            optimizer.step()

            batch_size = x_base.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        return {"train_loss": avg_loss}

    def evaluate(self, loader: DataLoader, criterion: Any) -> Dict[str, float]:
        """
        Evaluate the model on a validation or test set.

        Args:
            loader:    DataLoader yielding batches of (x_base, x_mid, x_slow, macro, y)
            criterion: Loss function.

        Returns:
            Dictionary with validation loss and, when scikit‑learn is available,
            ROC‑AUC score.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for evaluation.")
        self.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)):
                    raise ValueError("Batch must be a tuple/list.")
                x_base, x_mid, x_slow, macro, y = batch

                preds = self(x_base, x_mid, x_slow, macro)
                loss = criterion(preds, y.float())

                batch_size = x_base.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_preds.append(preds.detach())
                all_targets.append(y.detach())

        avg_loss = total_loss / max(total_samples, 1)
        metrics: Dict[str, float] = {"val_loss": avg_loss}

        if _HAS_SKLEARN:
            preds_concat = torch.cat(all_preds).cpu().numpy()
            targets_concat = torch.cat(all_targets).cpu().numpy()
            try:
                auc = roc_auc_score(targets_concat, preds_concat)
                metrics["val_auc"] = float(auc)
            except ValueError:
                # AUC cannot be computed (e.g., only one class present)
                metrics["val_auc"] = float("nan")

        return metrics


# ---------------------------------------------------------------------------
# Async training entry point
# ---------------------------------------------------------------------------

async def train(
    model: MultiScaleTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Any,
    criterion: Any,
    epochs: int = 10,
) -> Tuple[Dict[str, float], MultiScaleTransformer]:
    """
    Asynchronous training loop.

    Args:
        model:        Instance of MultiScaleTransformer.
        train_loader: DataLoader for the training set.
        val_loader:   DataLoader for the validation set.
        optimizer:    Optimizer instance.
        criterion:    Loss function.
        epochs:       Number of epochs to train.

    Returns:
        A tuple containing the best validation metrics dictionary and the
        model with the best validation state loaded.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training.")
    best_val_loss = float("inf")
    best_state = None
    best_metrics: Dict[str, float] = {}

    for epoch in range(epochs):
        train_metrics = model.train_epoch(train_loader, optimizer, criterion)
        val_metrics = model.evaluate(val_loader, criterion)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics

        # Optional: yield control to the event loop
        await asyncio.sleep(0)

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_metrics, model

# End of file