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

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]
    DataLoader = None   # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

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
    # dt: (B, T, d_inner) → (B, T, d_inner, 1)
    # A:  (d_inner, d_state) → (1, 1, d_inner, d_state)
    dt_expanded = dt.unsqueeze(-1)                       # (B, T, d_inner, 1)
    A_bar = torch.exp(dt_expanded * A[None, None, :, :]) # (B, T, d_inner, d_state)

    # Discretise B: B̄_t = Δ_t[:, :, :, None] * B_t[:, :, None, :]
    # B: (B, T, d_state) → (B, T, 1, d_state)
    B_bar = dt_expanded * B.unsqueeze(2)                  # (B, T, d_inner, d_state)

    # x expanded for SSM update: (B, T, d_inner) → (B, T, d_inner, 1)
    x_exp = x.unsqueeze(-1)                              # (B, T, d_inner, 1)

    # Sequential scan: h starts at zero
    h = x.new_zeros(B_sz, d_inner, d_state)             # (B, d_inner, d_state)
    ys = []
    for t in range(T):
        # h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
        h = A_bar[:, t, :, :] * h + B_bar[:, t, :, :] * x_exp[:, t, :, :]
        # y_t = sum over d_state of C_t * h_t  →  (B, d_inner)
        # C: (B, T, d_state) → C[:, t, :]: (B, d_state) → (B, 1, d_state)
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)     # (B, d_inner)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)                           # (B, T, d_inner)

    # Skip connection
    y = y + x * D[None, None, :]                        # (B, T, d_inner)
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
        # We use padding=d_conv-1 on the left and strip the extra right frames
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
        A = torch.arange(1, self.d_inner + 1, dtype=torch.float32).unsqueeze(1).expand(
            self.d_inner, d_state
        )
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)
        self.A_log._no_weight_decay = True        # type: ignore[attr-defined]

        # Skip connection scalar per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True            # type: ignore[attr-defined]

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
        x_in = self.norm(residual)             # Pre-LN

        # Split into gate z and main branch x
        projected = self.in_proj(x_in)         # (B, T, 2·d_inner)
        x, z = projected.chunk(2, dim=-1)      # each (B, T, d_inner)

        # Causal depthwise conv: (B, d_inner, T)
        x_conv = x.permute(0, 2, 1)            # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)            # (B, d_inner, T + d_conv - 1)
        x_conv = x_conv[:, :, :T]              # strip right padding → (B, d_inner, T)
        x = F.silu(x_conv.permute(0, 2, 1))    # (B, T, d_inner)

        # Compute SSM parameters from x
        ssm_params = self.x_proj(x)             # (B, T, 2·d_state + 1)
        B_raw = ssm_params[:, :, :self.d_state]                      # (B, T, d_state)
        C_raw = ssm_params[:, :, self.d_state:2 * self.d_state]      # (B, T, d_state)
        delta_raw = ssm_params[:, :, 2 * self.d_state:]              # (B, T, 1)

        # Δ (softplus): project 1 → d_inner
        dt = F.softplus(self.dt_proj(delta_raw))  # (B, T, d_inner)

        # A = -exp(A_log), always negative → stable discretisation
        A = -torch.exp(self.A_log.float())        # (d_inner, d_state)

        # Sequential selective scan
        y = selective_scan(x, dt, A, B_raw, C_raw, self.D)  # (B, T, d_inner)

        # Gate with SiLU(z)
        y = y * F.silu(z)                          # (B, T, d_inner)

        # Project back to d_model and add residual
        y = self.dropout(self.out_proj(y))         # (B, T, d_model)
        return residual + y


# ---------------------------------------------------------------------------
# MambaTrader
# ---------------------------------------------------------------------------

class MambaTrader(AbstractModel, nn.Module):
    """
    Mamba SSM model for trading signal prediction.

    Stacks N MambaBlocks with Pre-LN; mean-pools over time for classification.
    """
    model_type = "mamba_trader"

    def __init__(
        self,
        n_features: int = 27,
        seq_len: int = 64,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Stack of Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final norm + classification head
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch,) — raw logits (apply sigmoid for probabilities)

        Processing is strictly causal: each time step only attends to past
        steps via the sequential SSM scan — no lookahead.
        """
        # Project features to model dimension
        x = self.input_proj(x)     # (B, T, d_model)

        # Pass through Mamba blocks
        for block in self.blocks:
            x = block(x)           # (B, T, d_model)

        # Mean pool over the time dimension (uses all past information)
        x = x.mean(dim=1)          # (B, d_model)
        x = self.out_norm(x)
        logits = self.head(x).squeeze(-1)  # (B,)
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
    experiment_name: str = "mamba_trader_default",
    d_model: int = 128,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    n_layers: int = 4,
    dropout: float = 0.1,
    seq_len: int = 64,
    max_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
) -> dict:
    """
    Train a MambaTrader model on an OHLCV DataFrame.

    Returns a results dict with loss, accuracy, and artifact_path.
    Temporal (walk-forward) split is enforced with shuffle=False.
    """
    from app.ml.features.engineer import add_labels, create_sequences, engineer_features
    from app.ml.training.trainer import ARTIFACTS_DIR, train_with_lightning

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

    model = MambaTrader(
        n_features=n_features,
        seq_len=seq_len,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
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
        "seq_len": seq_len,
        "d_model": d_model,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand": expand,
        "n_layers": n_layers,
        "dropout": dropout,
        "experiment": experiment_name,
    }, str(save_path))

    results["artifact_path"] = str(save_path)
    return results
