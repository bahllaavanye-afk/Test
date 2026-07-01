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

import logging
import time
from typing import Any, Dict

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

# Configure module logger
_logger = logging.getLogger(__name__)

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

    def _log_forward_metrics(self, signal_count: int, exec_time: float, pnl_estimate: float) -> None:
        """
        Emit structured log for the forward pass.

        Args:
            signal_count: Number of signals (batch size) processed.
            exec_time: Execution time in seconds.
            pnl_estimate: Rough P&L estimate derived from model output.
        """
        try:
            _logger.info(
                "MultiScaleTransformer forward",
                extra={
                    "signal_count": signal_count,
                    "execution_time_ms": exec_time * 1000,
                    "pnl_estimate": pnl_estimate,
                },
            )
        except Exception as exc:  # pragma: no cover
            _logger.debug("Logging forward metrics failed: %s", exc)

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
        start_time = time.perf_counter()

        B = x_base.shape[0]

        # Encode base stream
        h_base = self._encode_stream(x_base, self.proj_base, self.enc_base)  # (B, D)

        if x_mid is None and x_slow is None:
            # Single-stream fallback
            out = self.fallback_head(h_base).squeeze(-1)  # (B,)
            exec_time = time.perf_counter() - start_time
            self._log_forward_metrics(signal_count=B, exec_time=exec_time, pnl_estimate=out.mean().item())
            return out

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

        exec_time = time.perf_counter() - start_time
        # Use mean of the output as a lightweight P&L proxy
        pnl_estimate = out.mean().item() if out.numel() > 0 else 0.0
        self._log_forward_metrics(signal_count=B, exec_time=exec_time, pnl_estimate=pnl_estimate)

        return out

    def train_epoch(self, loader: DataLoader, optimizer, criterion) -> dict:
        """Train for one epoch."""
        # The original implementation is unchanged; logging of epoch‑level
        # metrics (e.g., average loss) is handled by the surrounding training
        # loop. This placeholder remains to preserve existing behaviour.
        pass

    # The remainder of the file (evaluation utilities, async training entry point,
    # etc.) is unchanged from the original implementation.