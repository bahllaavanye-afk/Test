"""
Lorentzian KNN Classifier — Python port of jdehorty's TradingView indicator.

Lorentzian distance is more robust to outliers than Euclidean:
  d(x, y) = sqrt(sum(log(1 + |xi - yi|)^2))

This handles black‑swan events better because the log function
compresses extreme differences, preventing rare events from dominating.

Features used (same as original TV indicator):
  RSI(14), CCI(20), ADX(20), EMA delta (fast vs slow), SMA delta
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import app.ml.features.pandas_ta_compat as ta
from app.ml.models.base_model import AbstractModel, EvalMetrics

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LORENTZIAN_FEATURES = [
    "rsi_14",
    "cci_20",
    "adx_20",
    "ema_fast_delta",
    "ema_slow_delta",
]

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def lorentzian_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Lorentzian distance between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(..., n_features)``.
    y : torch.Tensor
        Shape ``(..., n_features)`` – must be broadcast‑compatible with ``x``.

    Returns
    -------
    torch.Tensor
        Distance with shape ``(...)``.
    """
    diff = torch.abs(x - y)
    # log1p is numerically stable for small values
    log_term = torch.log1p(diff)
    return torch.sqrt(torch.sum(log_term ** 2, dim=-1))


def compute_lorentzian_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the five features used by the Lorentzian classifier.

    The function is side‑effect free – it works on a copy of ``df`` and
    returns a new DataFrame with the required columns added.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least a ``close`` column. ``high`` and ``low`` are optional.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional feature columns.
    """
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)

    out = df.copy()

    # RSI – scaled to [0, 1]
    rsi = ta.rsi(close, length=14)
    out["rsi_14"] = (rsi / 100.0) if rsi is not None else 0.5

    # CCI – scaled to [-1, 1] with clipping
    cci = ta.cci(high, low, close, length=20)
    out["cci_20"] = (cci / 200.0).clip(-1, 1) if cci is not None else 0.0

    # ADX – scaled to [0, 1]
    adx_df = ta.adx(high, low, close, length=20)
    if adx_df is not None and "ADX_20" in adx_df.columns:
        out["adx_20"] = adx_df["ADX_20"] / 100.0
    else:
        out["adx_20"] = 0.5

    # EMA deltas – normalised by price to keep values bounded
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()

    eps = 1e-9
    out["ema_fast_delta"] = (ema_fast - ema_slow) / (close + eps)
    out["ema_slow_delta"] = (ema_slow - ema_200) / (close + eps)

    return out


# --------------------------------------------------------------------------- #
# Model definition
# --------------------------------------------------------------------------- #


class LorentzianKNN(AbstractModel):
    """
    K‑Nearest‑Neighbors classifier using Lorentzian distance.

    The model keeps a fixed‑size library of historical feature vectors
    and performs a nearest‑neighbour lookup at inference time.
    """

    model_type = "lorentzian_knn"

    def __init__(self, k: int = 8, lookback: int = 2000, subsample: int = 4) -> None:
        self.k = k
        self.lookback = lookback
        self.subsample = subsample
        self._library_X: torch.Tensor | None = None
        self._library_y: torch.Tensor | None = None

    # --------------------------------------------------------------------- #
    # Core inference
    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities for a batch of feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)`` – mean label of the ``k`` nearest neighbours.
        """
        if self._library_X is None or self._library_y is None:
            # No library – return a neutral probability
            return torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)

        # Vectorised Lorentzian distance computation
        # x: (B, n) → (B, 1, n)
        # library_X: (N, n) → (1, N, n)
        diff = x.unsqueeze(1) - self._library_X.unsqueeze(0)  # (B, N, n)
        dists = torch.sqrt(torch.sum(torch.log1p(torch.abs(diff)) ** 2, dim=2))  # (B, N)

        # Retrieve indices of the k nearest neighbours (smallest distances)
        _, top_k = torch.topk(dists, self.k, dim=1, largest=False)

        # Gather neighbour labels and compute the mean
        neighbour_labels = self._library_y[top_k]  # (B, k)
        probs = neighbour_labels.float().mean(dim=1)  # (B,)

        return probs

    # --------------------------------------------------------------------- #
    # Library management
    # --------------------------------------------------------------------- #

    def fit_library(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Populate the KNN library from training data.

        Sub‑samples the data according to ``self.subsample`` and keeps only
        the most recent ``self.lookback`` entries.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(samples, n_features)``.
        y : np.ndarray
            Target vector of shape ``(samples,)``.
        """
        idx = np.arange(0, len(X), self.subsample, dtype=int)
        if len(idx) > self.lookback:
            idx = idx[-self.lookback :]
        self._library_X = torch.tensor(X[idx], dtype=torch.float32)
        self._library_y = torch.tensor(y[idx], dtype=torch.float32)

    # --------------------------------------------------------------------- #
    # Training / evaluation hooks (required by AbstractModel)
    # --------------------------------------------------------------------- #

    def train_epoch(self, loader: Iterable[Any], optimizer: Any = None, criterion: Any = None) -> dict:
        # KNN has no trainable parameters – nothing to optimise.
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> EvalMetrics:
        """
        Compute accuracy and AUC over a validation loader.
        """
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for X_batch, y_batch in loader:
            probs = self.forward(X_batch).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(y_batch.cpu().numpy())

        probs_cat = np.concatenate(all_preds)
        labels_cat = np.concatenate(all_labels)

        preds = (probs_cat > 0.5).astype(int)
        acc = float((preds == labels_cat).mean())

        try:
            auc = float(roc_auc_score(labels_cat, probs_cat))
        except ValueError:
            auc = 0.5  # fallback when only one class is present

        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #

    def save(self, path: str, metadata: Dict[str, Any] | None = None) -> None:
        """
        Serialise the model to ``path`` using pickle.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "library_X": self._library_X,
            "library_y": self._library_y,
            "k": self.k,
            "lookback": self.lookback,
            "model_type": self.model_type,
            "metadata": metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "LorentzianKNN":
        """
        Load a model saved with :meth:`save`.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(k=data["k"], lookback=data["lookback"])
        model._library_X = data["library_X"]
        model._library_y = data["library_y"]
        return model

    # --------------------------------------------------------------------- #
    # Signal generation utilities (strategy‑level helpers)
    # --------------------------------------------------------------------- #

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Alias for ``forward`` – kept for semantic clarity.
        """
        return self.forward(X)

    def generate_signal(
        self,
        X: torch.Tensor,
        entry_thresh: float = 0.6,
        exit_thresh: float = 0.4,
        consensus_std: float = 0.1,
    ) -> torch.Tensor:
        """
        Produce trading signals based on neighbour consensus.

        - **Entry (1)** when the predicted probability exceeds ``entry_thresh``
          and the standard deviation of neighbour labels is below ``consensus_std``.
        - **Exit (-1)** when the probability falls below ``exit_thresh`` with the same
          consensus requirement.
        - **Hold (0)** otherwise.

        Parameters
        ----------
        X : torch.Tensor
            Input feature batch.
        entry_thresh : float
            Minimum probability for a long entry.
        exit_thresh : float
            Maximum probability for a short exit.
        consensus_std : float
            Maximum allowed standard deviation among the k neighbour labels.

        Returns
        -------
        torch.Tensor
            Signal tensor of shape ``(batch,)`` with values in ``{-1, 0, 1}``.
        """
        if self._library_X is None or self._library_y is None:
            return torch.zeros(X.shape[0], dtype=torch.int8, device=X.device)

        # Compute distances and obtain neighbour indices (vectorised)
        diff = X.unsqueeze(1) - self._library_X.unsqueeze(0)  # (B, N, n)
        dists = torch.sqrt(torch.sum(torch.log1p(torch.abs(diff)) ** 2, dim=2))
        _, top_k = torch.topk(dists, self.k, dim=1, largest=False)

        neighbour_labels = self._library_y[top_k]  # (B, k)

        probs = neighbour_labels.float().mean(dim=1)  # (B,)
        stds = neighbour_labels.float().std(dim=1)  # (B,)

        # Apply thresholds
        entry = (probs >= entry_thresh) & (stds <= consensus_std)
        exit_ = (probs <= exit_thresh) & (stds <= consensus_std)

        signal = torch.zeros_like(probs, dtype=torch.int8)
        signal[entry] = 1
        signal[exit_] = -1
        return signal