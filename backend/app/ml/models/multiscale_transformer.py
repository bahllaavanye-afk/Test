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
from app.ml.models.patch_tst import PatchEncoder


# ---------------------------------------------------------------------------
# Cross-Attention module
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Single-head cross-attention: Q from one stream, K/V from another.
    Both inputs are (batch, d_model) vectors; we add a sequence dimension of 1
    so standard MultiheadAttention is used directly.
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
            (batch, d_model) — cross-attended query
        """
        q = query.unsqueeze(1)   # (B, 1, D)
        k = kv.unsqueeze(1)      # (B, 1, D)
        out, _ = self.attn(q, k, k)  # (B, 1, D)
        out = out.squeeze(1)     # (B, D)
        return self.norm(out + query)  # residual + norm


# ---------------------------------------------------------------------------
# MultiScaleTransformer
# ---------------------------------------------------------------------------

class MultiScaleTransformer(AbstractModel, nn.Module):
    """
    Three-stream cross-attention transformer for multi-resolution trading signals.
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
        # base stream: seq=60, patch=8
        self.enc_base = PatchEncoder(seq_len=60, patch_len=8, d_model=d_model,
                                     n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        # mid stream: seq=20, patch=4
        self.enc_mid = PatchEncoder(seq_len=20, patch_len=4, d_model=d_model,
                                    n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        # slow stream: seq=10, patch=2
        self.enc_slow = PatchEncoder(seq_len=10, patch_len=2, d_model=d_model,
                                     n_heads=n_heads, n_layers=n_layers, dropout=dropout)

        # --- Channel-independent projections (for multi-feature inputs) ---
        # Each PatchEncoder expects single-channel (batch, seq_len).
        # We average across feature channels before encoding.
        # Alternatively, project features → 1 per stream.
        self.proj_base = nn.Linear(n_features_base, 1)
        self.proj_mid = nn.Linear(n_features_mid, 1)
        self.proj_slow = nn.Linear(n_features_slow, 1)

        # --- Cross-attention layers ---
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

        # --- Single-stream fallback head ---
        self.fallback_head = nn.Linear(d_model, 1)

    def _encode_stream(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
        encoder: PatchEncoder,
    ) -> torch.Tensor:
        """
        Encode a (batch, seq_len, n_features) tensor using channel projection
        followed by PatchEncoder.

        Returns: (batch, d_model)
        """
        # Project features to single channel: (B, T, 1) → (B, T)
        x_1d = proj(x).squeeze(-1)   # (B, T)
        return encoder(x_1d)          # (B, d_model)

    def forward(
        self,
        x_base: torch.Tensor,
        x_mid: torch.Tensor | None = None,
        x_slow: torch.Tensor | None = None,
        macro: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_base:  (batch, 60, n_features_base)
            x_mid:   (batch, 20, n_features_mid)   or None
            x_slow:  (batch, 10, n_features_slow)  or None
            macro:   (batch, n_macro)               or None

        Returns:
            (batch,)  — logits
        """
        B = x_base.shape[0]

        # Encode base stream
        h_base = self._encode_stream(x_base, self.proj_base, self.enc_base)  # (B, D)

        if x_mid is None and x_slow is None:
            # Single-stream fallback
            return self.fallback_head(h_base).squeeze(-1)  # (B,)

        # Encode mid and slow streams
        if x_mid is not None:
            h_mid = self._encode_stream(x_mid, self.proj_mid, self.enc_mid)    # (B, D)
        else:
            h_mid = torch.zeros(B, self.d_model, device=x_base.device, dtype=x_base.dtype)

        if x_slow is not None:
            h_slow = self._encode_stream(x_slow, self.proj_slow, self.enc_slow)  # (B, D)
        else:
            h_slow = torch.zeros(B, self.d_model, device=x_base.device, dtype=x_base.dtype)

        # Cross-attention fusion
        h12 = self.cross_base_mid(h_base, h_mid)    # (B, D)
        h123 = self.cross_12_slow(h12, h_slow)       # (B, D)

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

    def train_epoch(self, loader: DataLoader, optimizer, criterion) -> dict:
        """Train for one epoch on single-stream data (x_base only)."""
        self.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                X, y = batch
            else:
                raise ValueError("Loader must yield (X, y) pairs.")

            optimizer.zero_grad()
            # Single-stream mode when loader yields (B, T, C)
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
        """Evaluate model on a DataLoader using single-stream mode."""
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
# Training entry point
# ---------------------------------------------------------------------------

async def train(
    ohlcv_df,
    experiment_name: str = "multiscale_transformer_default",
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.2,
    seq_len: int = 60,
    max_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
) -> dict:
    """
    Train MultiScaleTransformer in single-stream mode on OHLCV data.
    Returns a results dict with loss, accuracy, and artifact_path.
    """
    import torch
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

    model = MultiScaleTransformer(
        n_features_base=n_features,
        n_features_mid=n_features,
        n_features_slow=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
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
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "seq_len": seq_len,
        "experiment": experiment_name,
    }, str(save_path))

    results["artifact_path"] = str(save_path)
    return results
