"""
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.
Wu et al. (ICLR 2023) — https://arxiv.org/abs/2210.02186

Key insight: transform 1D time series into 2D space by reshaping based on
dominant periods detected via FFT. Applies 2D convolutions to capture
intra-period (trend) and inter-period (seasonality) variation simultaneously.

For trading signals:
  - Input: (batch, seq_len, n_features) OHLCV + technical indicators
  - Output: (batch, 1) — probability of upward price movement

Advantages over LSTM for finance:
  - Captures multi-scale periodicity (daily/weekly/monthly cycles simultaneously)
  - 2D Conv naturally models price pattern repetition
  - More interpretable: FFT shows which frequencies drive the signal
  - State-of-the-art on financial forecasting benchmarks (2023)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from typing import Optional


def _top_k_periods(x: torch.Tensor, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Detect top-k dominant periods in the input via FFT.
    x: (batch, seq_len, n_vars)
    Returns: (amplitudes, periods) each of shape (k,)
    """
    # Average over batch and variables for period detection
    xf = torch.fft.rfft(x.mean(dim=(0, 2)))  # (freq_bins,)
    freqs = torch.fft.rfftfreq(x.shape[1], device=x.device)

    # Amplitudes (magnitude of FFT)
    amp = torch.abs(xf)
    amp[0] = 0  # zero out DC component

    # Top-k frequencies by amplitude (skip DC)
    _, top_k_idx = torch.topk(amp[1:], k=min(k, len(amp) - 1))
    top_k_idx = top_k_idx + 1  # re-add the offset

    top_freqs = freqs[top_k_idx]
    # Convert frequency → period (in timesteps)
    periods = (1.0 / (top_freqs + 1e-8)).long().clamp(2, x.shape[1])

    return amp[top_k_idx], periods


class TimesBlock(nn.Module):
    """
    Core TimesNet block.
    1. Reshape 1D → 2D using detected period p: (batch, n_vars, p, seq//p)
    2. Apply 2D inception conv to capture 2D variation
    3. Reshape back and aggregate
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        top_k: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.top_k = top_k
        padding = kernel_size // 2

        # Inception-style 2D Conv (multi-scale)
        self.conv1 = nn.Sequential(
            nn.Conv2d(d_model, d_ff, (1, kernel_size), padding=(0, padding)),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, (1, 1)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(d_model, d_ff, (3, kernel_size), padding=(1, padding)),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, (1, 1)),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        B, T, C = x.shape
        _, periods = _top_k_periods(x, k=self.top_k)

        res_list = []
        for p in periods:
            p = max(int(p.item()), 2)
            # Pad to make T divisible by p
            pad_len = (p - T % p) % p
            x_pad = F.pad(x.permute(0, 2, 1), (0, pad_len))  # (B, C, T+pad)
            t_padded = T + pad_len
            n_periods = t_padded // p

            # Reshape to 2D: (B, C, n_periods, p)
            x_2d = x_pad.reshape(B, C, n_periods, p)

            # 2D conv (inception-style)
            if n_periods >= 3:
                out_2d = self.conv3(x_2d)
            else:
                out_2d = self.conv1(x_2d)

            # Reshape back to 1D
            out_1d = out_2d.reshape(B, C, -1)[:, :, :T]  # trim padding
            res_list.append(out_1d.permute(0, 2, 1))  # (B, T, C)

        # Aggregate by amplitude-weighted average
        if len(res_list) == 1:
            res = res_list[0]
        else:
            res = torch.stack(res_list, dim=0).mean(dim=0)

        return self.norm(x + self.dropout(res))


class TimesNetPredictor(nn.Module):
    """
    Full TimesNet model for binary trading signal prediction.

    Architecture:
      - Linear embedding: n_features → d_model
      - N TimesBlocks with 2D temporal variation modeling
      - Global average pooling over time
      - Linear head → sigmoid probability

    Same interface as LSTMPredictor and NBEATSPredictor.
    """

    def __init__(
        self,
        input_size: int = 30,       # number of input features
        seq_len: int = 60,          # lookback window
        d_model: int = 64,
        d_ff: int = 128,
        n_layers: int = 3,
        top_k: int = 3,             # top-k periods from FFT
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(d_model=d_model, d_ff=d_ff, top_k=top_k)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        Returns: (batch, 1) probability [0, 1]
        """
        x = self.embedding(x)          # (B, T, d_model)
        for block in self.blocks:
            x = block(x)               # (B, T, d_model)
        x = x.mean(dim=1)              # global average pool → (B, d_model)
        return self.head(x)            # (B, 1)


class TimesNetWrapper:
    """
    Scikit-learn-style wrapper for TimesNetPredictor.
    Handles training, evaluation, save/load.
    Matches the interface used by ensemble_model.py.
    """

    MODEL_ID = "timesnet"

    def __init__(
        self,
        input_size: int = 30,
        seq_len: int = 60,
        d_model: int = 64,
        d_ff: int = 128,
        n_layers: int = 3,
        top_k: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimesNetPredictor(
            input_size=input_size,
            seq_len=seq_len,
            d_model=d_model,
            d_ff=d_ff,
            n_layers=n_layers,
            top_k=top_k,
            dropout=dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.BCELoss()

    def train_epoch(self, loader) -> dict:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device).float()
            self.optimizer.zero_grad()
            preds = self.model(X).squeeze(-1)
            loss = self.criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y)
            correct += ((preds > 0.5) == (y > 0.5)).sum().item()
            total += len(y)
        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device).float()
            preds = self.model(X).squeeze(-1)
            loss = self.criterion(preds, y)
            total_loss += loss.item() * len(y)
            correct += ((preds > 0.5) == (y > 0.5)).sum().item()
            total += len(y)
        return {
            "val_loss": total_loss / max(total, 1),
            "val_accuracy": correct / max(total, 1),
        }

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> float:
        """Single-sample inference. x: (seq_len, n_features)"""
        self.model.eval()
        x = x.unsqueeze(0).to(self.device)
        return float(self.model(x).item())

    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "metadata": metadata or {},
            "model_id": self.MODEL_ID,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TimesNetWrapper":
        data = torch.load(path, map_location="cpu", weights_only=False)
        meta = data.get("metadata", {})
        wrapper = cls(
            input_size=meta.get("input_size", 30),
            seq_len=meta.get("seq_len", 60),
            d_model=meta.get("d_model", 64),
            d_ff=meta.get("d_ff", 128),
            n_layers=meta.get("n_layers", 3),
            top_k=meta.get("top_k", 3),
        )
        wrapper.model.load_state_dict(data["state_dict"])
        return wrapper
