"""
MambaTrader — Selective State Space Model (Mamba / S6, Gu & Dao, NeurIPS 2023).

Pure PyTorch implementation; no mamba-ssm CUDA package required.
Sequences are capped at 500 steps, so the sequential scan is fast enough
without a custom CUDA kernel.

Core S6 (Selective SSM) mechanics per time step t:
  h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ (x_t expanded)
  y_t = C_t · h_t  +  D ⊙ x_t          (skip connection)

where the discretisation Ā_t / B̄_t and the inputs B_t, C_t, Δ_t are all
*input-dependent* (hence "selective").

Architecture:
  Input projection: n_features → d_model
  N × MambaBlock (each block):
    LayerNorm (Pre-LN)
    in_proj: d_model → 2·d_inner  →  z (gate), x (main branch)
    conv1d (depthwise, kernel=d_conv, padding=d_conv-1): d_inner → d_inner
    x_proj: d_inner → (d_state + d_state + 1)  →  B_raw, C_raw, Δ_raw
    dt_proj: 1 → d_inner  (Δ after softplus)
    A_log: (d_inner, d_state)  HiPPO-style: log(-A)
    D:     (d_inner,)           skip
    SSM sequential scan → y: (B, T, d_inner)
    Gate: y = y ⊙ SiLU(z)
    out_proj: d_inner → d_model
    Residual: output + input
  Mean-pool over time → LayerNorm → Linear(d_model, 1) → squeeze → (B,)
  (raw logits; BCEWithLogitsLoss in training, sigmoid in predict_proba)

Exports:
  MambaTrader   — model class
  train(...)    — async training entry point matching train_lstm.py API
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from app.ml.models.base_model import AbstractModel, EvalMetrics


# ---------------------------------------------------------------------------
# Selective SSM scan  (sequential; causal — no lookahead)
# ---------------------------------------------------------------------------


def selective_scan(
    x: torch.Tensor,   # (B, T, d_inner)
    dt: torch.Tensor,  # (B, T, d_inner)
    A: torch.Tensor,   # (d_inner, d_state)  negative values
    B: torch.Tensor,   # (B, T, d_state)
    C: torch.Tensor,   # (B, T, d_state)
    D: torch.Tensor,   # (d_inner,)
) -> torch.Tensor:
    """
    Sequential causal scan implementing the S6 recurrence:
      Ā_t = exp(Δ_t ⊗ A)              shape: (B, d_inner, d_state)
      B̄_t = Δ_t ⊗ B_t                shape: (B, d_inner, d_state)
      h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t[:, :, :, None]
      y_t = (C_t · h_t).sum(-1)  +  D ⊙ x_t

    Returns:
        y: (B, T, d_inner)

    Note: loop over T is O(T·B·d_inner·d_state). For T≤500 this is fast enough
    on CPU/GPU without a fused kernel.
    """
    B_sz, T, d_inner = x.shape
    d_state = A.shape[1]

    # Discretise A: Ā_t = exp(Δ_t[:, :, :, None] * A[None, None, :, :])
    dt_expanded = dt.unsqueeze(-1)                       # (B, T, d_inner, 1)
    A_bar = torch.exp(dt_expanded * A[None, None, :, :])  # (B, T, d_inner, d_state)

    # Discretise B: B̄_t = Δ_t[:, :, :, None] * B_t[:, :, None, :]
    B_bar = dt_expanded * B.unsqueeze(2)                  # (B, T, d_inner, d_state)

    # x expanded for SSM update: (B, T, d_inner) → (B, T, d_inner, 1)
    x_exp = x.unsqueeze(-1)                               # (B, T, d_inner, 1)

    # Sequential scan: h starts at zero
    h = x.new_zeros(B_sz, d_inner, d_state)               # (B, d_inner, d_state)
    ys = []
    for t in range(T):
        h = A_bar[:, t, :, :] * h + B_bar[:, t, :, :] * x_exp[:, t, :, :]
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)       # (B, d_inner)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)                           # (B, T, d_inner)

    # Skip connection
    y = y + x * D[None, None, :]                         # (B, T, d_inner)
    return y


# ---------------------------------------------------------------------------
# Mamba Block
# ---------------------------------------------------------------------------


class MambaBlock(nn.Module):
    """
    One Mamba (S6) block with Pre-LN, gating, and residual connection.

    Input / output: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)

        # Expand input to z (gate) + x (main) branches
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise 1-D conv for local context (causal: keep only left context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # Project to SSM parameters: B (d_state), C (d_state), Δ (1)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1, bias=False)

        # Δ projection: 1 → d_inner
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # HiPPO-style A initialisation: A_{n,k} = -(n+1)  (diagonal approx)
        # Stored as log(-A) so A = -exp(A_log) < 0 always
        A_init = torch.arange(1, self.d_inner + 1, dtype=torch.float32).unsqueeze(1).expand(
            self.d_inner, d_state
        )
        self.A_log = nn.Parameter(torch.log(A_init))  # (d_inner, d_state)
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # Skip connection scalar per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) — output + residual
        """
        B, T, _ = residual.shape
        x_in = self.norm(residual)                     # Pre-LN

        # Split into gate z and main branch x
        projected = self.in_proj(x_in)                 # (B, T, 2·d_inner)
        x, z = projected.chunk(2, dim=-1)              # each (B, T, d_inner)

        # Causal depthwise conv: (B, d_inner, T)
        x_conv = x.permute(0, 2, 1)                     # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)                   # (B, d_inner, T + d_conv - 1)
        x_conv = x_conv[:, :, :T]                       # strip right padding → (B, d_inner, T)
        x = F.silu(x_conv.permute(0, 2, 1))             # (B, T, d_inner)

        # Compute SSM parameters from x
        ssm_params = self.x_proj(x)                    # (B, T, 2·d_state + 1)
        B_raw = ssm_params[:, :, :self.d_state]                       # (B, T, d_state)
        C_raw = ssm_params[:, :, self.d_state:2 * self.d_state]       # (B, T, d_state)
        delta_raw = ssm_params[:, :, 2 * self.d_state:]               # (B, T, 1)

        # Δ (softplus) → (B, T, d_inner)
        dt = F.softplus(self.dt_proj(delta_raw))

        # A is constant across batch & time
        A = -torch.exp(self.A_log)                     # (d_inner, d_state)

        # SSM scan
        y = selective_scan(x, dt, A, B_raw, C_raw, self.D)

        # Gating
        y = y * F.silu(z)

        # Output projection, dropout, and residual addition
        y = self.out_proj(y)
        y = self.dropout(y)
        return y + residual


# ---------------------------------------------------------------------------
# MambaTrader Model
# ---------------------------------------------------------------------------


class MambaTrader(AbstractModel):
    """
    MambaTrader model composed of an input projection, a stack of MambaBlocks,
    and a final classification head.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_blocks: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # mean over time dimension
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch,).
        """
        h = self.input_proj(x)            # (B, T, d_model)
        for block in self.blocks:
            h = block(h)                  # (B, T, d_model)
        # Mean pooling over the time dimension
        h = h.permute(0, 2, 1)            # (B, d_model, T