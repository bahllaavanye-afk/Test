"""
Lorentzian KNN Classifier — Python port of jdehorty's TradingView indicator.

Lorentzian distance is more robust to outliers than Euclidean:
  d(x, y) = sqrt(sum(log(1 + |xi - yi|)^2))

This handles black-swan events better because the log function
compresses extreme differences, preventing rare events from dominating.

Features used (same as original TV indicator):
  RSI(14), CCI(20), ADX(20), EMA delta (fast vs slow), SMA delta
"""
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]

import numpy as np
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from sklearn.metrics import roc_auc_score
from app.ml.models.base_model import AbstractModel, EvalMetrics

LORENTZIAN_FEATURES = ["rsi_14", "cci_20", "adx_20", "ema_fast_delta", "ema_slow_delta"]


def lorentzian_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lorentzian distance between two feature vectors."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for lorentzian_distance — install with `pip install torch`"
        )
    if x is None or y is None:
        raise ValueError("Both x and y must be provided for lorentzian_distance.")
    if x.shape != y.shape:
        raise ValueError(
            f"Shape mismatch: x.shape={x.shape} vs y.shape={y.shape} in lorentzian_distance."
        )
    diff = torch.abs(x - y)
    # log(1 + |xi - yi|) is always defined; avoid domain errors
    return torch.sqrt(torch.sum(torch.log1p(diff) ** 2, dim=-1))


def compute_lorentzian_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 features used by the Lorentzian classifier."""
    if df is None or df.empty:
        # Return an empty DataFrame with expected columns to keep downstream code safe
        return pd.DataFrame(columns=LORENTZIAN_FEATURES)

    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)

    df = df.copy()
    rsi = ta.rsi(close, length=14)
    cci = ta.cci(high, low, close, length=20)
    adx_df = ta.adx(high, low, close, length=20)

    df["rsi_14"] = (rsi / 100.0) if rsi is not None else 0.5
    df["cci_20"] = (cci / 200.0).clip(-1, 1) if cci is not None else 0.0
    df["adx_20"] = (
        (adx_df["ADX_20"] / 100.0)
        if (adx_df is not None and "ADX_20" in adx_df.columns)
        else 0.5
    )

    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    df["ema_fast_delta"] = (ema_fast - ema_slow) / (close + 1e-9)
    df["ema_slow_delta"] = (ema_slow - ema_200) / (close + 1e-9)

    return df


class LorentzianKNN(AbstractModel):
    """
    KNN with Lorentzian distance. Stores historical feature library.
    k=8 neighbors, max lookback=2000 bars, subsampling every 4 bars.
    """
    model_type = "lorentzian_knn"

    def __init__(self, k: int = 8, lookback: int = 2000, subsample: int = 4):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for LorentzianKNN — install with `pip install torch`"
            )
        self.k = max(1, k)
        self.lookback = max(1, lookback)
        self.subsample = max(1, subsample)
        self._library_X = None  # type: ignore
        self._library_y = None  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_features) — single-step inference (no sequence)."""
        if x is None or x.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=x.device if x is not None else "cpu")
        if self._library_X is None or self._library_X.numel() == 0:
            return torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)

        results = []
        max_k = min(self.k, self._library_X.shape[0])
        for i in range(x.shape[0]):
            query = x[i].unsqueeze(0)  # (1, n_features)
            # Expand query to match library size for vectorized distance computation
            dists = lorentzian_distance(
                query.expand_as(self._library_X), self._library_X
            )
            _, top_k = torch.topk(dists, max_k, largest=False)
            k_labels = self._library_y[top_k].float()
            results.append(k_labels.mean())
        return torch.stack(results)

    def fit_library(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the KNN library from training data (subsampled)."""
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            self._library_X = None
            self._library_y = None
            return

        idx = np.arange(0, len(X), self.subsample)
        if len(idx) > self.lookback:
            idx = idx[-self.lookback :]
        self._library_X = torch.tensor(X[idx], dtype=torch.float32)
        self._library_y = torch.tensor(y[idx], dtype=torch.float32)

    def train_epoch(self, loader, optimizer=None, criterion=None) -> dict:
        # KNN has no training loop — fit_library is called directly
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        all_preds, all_labels = [], []
        for X, y in loader:
            if X is None or y is None or X.numel() == 0:
                continue
            probs = self.forward(X).numpy()
            all_preds.append(probs)
            all_labels.append(y.numpy())
        if not all_preds:
            return EvalMetrics(accuracy=0.0, auc=0.5, sharpe=0.0)

        probs_cat = np.concatenate(all_preds)
        labels_cat = np.concatenate(all_labels)
        preds = (probs_cat > 0.5).astype(int)
        acc = float((preds == labels_cat).mean())
        try:
            auc = float(roc_auc_score(labels_cat, probs_cat))
        except ValueError:
            auc = 0.5
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    def save(self, path: str, metadata: dict | None = None) -> None:
        import pickle
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "library_X": self._library_X,
                    "library_y": self._library_y,
                    "k": self.k,
                    "lookback": self.lookback,
                    "model_type": self.model_type,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "LorentzianKNN":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(k=data["k"], lookback=data["lookback"])
        model._library_X = data["library_X"]
        model._library_y = data["library_y"]
        return model