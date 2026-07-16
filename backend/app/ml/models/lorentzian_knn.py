"""
Lorentzian KNN Classifier — Python port of jdehorty's TradingView indicator.

Lorentzian distance is more robust to outliers than Euclidean:
  d(x, y) = sqrt(sum(log(1 + |xi - yi|)^2))

This handles black-swan events better because the log function
compresses extreme differences, preventing rare events from dominating.

Features used (same as original TV indicator):
  RSI(14), CCI(20), ADX(20), EMA delta (fast vs slow), SMA delta
"""

# Constants
RSI_LENGTH = 14
CCI_LENGTH = 20
ADX_LENGTH = 20
EMA_FAST_SPAN = 9
EMA_SLOW_SPAN = 21
EMA_LONG_SPAN = 200
EPSILON = 1e-9

DEFAULT_RSI_VALUE = 0.5
DEFAULT_CCI_VALUE = 0.0
DEFAULT_ADX_VALUE = 0.5

DEFAULT_K = 8
DEFAULT_LOOKBACK = 2000
DEFAULT_SUBSAMPLE = 4

FEATURE_NAMES = ["rsi_14", "cci_20", "adx_20", "ema_fast_delta", "ema_slow_delta"]
ADX_COLUMN_NAME = "ADX_20"

# Try to import torch; flag availability
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


def lorentzian_distance(x, y):
    """Lorentzian distance between two feature vectors."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for lorentzian_distance — install with `pip install torch`")
    return torch.sqrt(torch.sum(torch.log(1 + torch.abs(x - y)) ** 2, dim=-1))


def compute_lorentzian_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 features used by the Lorentzian classifier."""
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)

    df = df.copy()
    rsi = ta.rsi(close, length=RSI_LENGTH)
    cci = ta.cci(high, low, close, length=CCI_LENGTH)
    adx_df = ta.adx(high, low, close, length=ADX_LENGTH)

    df["rsi_14"] = (rsi / 100.0) if rsi is not None else DEFAULT_RSI_VALUE
    df["cci_20"] = (cci / 200.0).clip(-1, 1) if cci is not None else DEFAULT_CCI_VALUE
    df["adx_20"] = (adx_df[ADX_COLUMN_NAME] / 100.0) if (adx_df is not None and ADX_COLUMN_NAME in adx_df.columns) else DEFAULT_ADX_VALUE

    ema_fast = close.ewm(span=EMA_FAST_SPAN).mean()
    ema_slow = close.ewm(span=EMA_SLOW_SPAN).mean()
    ema_long = close.ewm(span=EMA_LONG_SPAN).mean()
    df["ema_fast_delta"] = (ema_fast - ema_slow) / (close + EPSILON)
    df["ema_slow_delta"] = (ema_slow - ema_long) / (close + EPSILON)

    return df


class LorentzianKNN(AbstractModel):
    """
    KNN with Lorentzian distance. Stores historical feature library.
    k=8 neighbors, max lookback=2000 bars, subsampling every 4 bars.
    """
    model_type = "lorentzian_knn"

    def __init__(self, k: int = DEFAULT_K, lookback: int = DEFAULT_LOOKBACK, subsample: int = DEFAULT_SUBSAMPLE):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for LorentzianKNN — install with `pip install torch`")
        self.k = k
        self.lookback = lookback
        self.subsample = subsample
        self._library_X = None  # type: ignore
        self._library_y = None  # type: ignore

    def forward(self, x):
        """x: (batch, n_features) — single-step inference (no sequence)."""
        if self._library_X is None:
            return torch.zeros(x.shape[0])

        results = []
        for i in range(x.shape[0]):
            query = x[i].unsqueeze(0)  # (1, n_features)
            dists = lorentzian_distance(query.expand_as(self._library_X), self._library_X)
            _, top_k = torch.topk(dists, self.k, largest=False)
            k_labels = self._library_y[top_k].float()
            results.append(k_labels.mean())
        return torch.stack(results)

    def fit_library(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the KNN library from training data (subsampled)."""
        idx = np.arange(0, len(X), self.subsample)
        if len(idx) > self.lookback:
            idx = idx[-self.lookback:]
        self._library_X = torch.tensor(X[idx], dtype=torch.float32)
        self._library_y = torch.tensor(y[idx], dtype=torch.float32)

    def train_epoch(self, loader, optimizer=None, criterion=None) -> dict:
        # KNN has no training loop — fit_library is called directly
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        all_preds, all_labels = [], []
        for X, y in loader:
            probs = self.forward(X).numpy()
            all_preds.append(probs)
            all_labels.append(y.numpy())
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
            pickle.dump({
                "library_X": self._library_X,
                "library_y": self._library_y,
                "k": self.k,
                "lookback": self.lookback,
                "model_type": self.model_type,
            }, f)

    @classmethod
    def load(cls, path: str) -> "LorentzianKNN":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(k=data["k"], lookback=data["lookback"])
        model._library_X = data["library_X"]
        model._library_y = data["library_y"]
        return model