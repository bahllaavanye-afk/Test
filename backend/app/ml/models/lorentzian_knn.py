"""
Lorentzian KNN Classifier — Python port of jdehorty's TradingView indicator.

This module implements a K‑Nearest Neighbours classifier that uses the
Lorentzian distance metric, which is more robust to outliers than the
standard Euclidean distance. The distance is defined as::

    d(x, y) = sqrt( sum( log(1 + |x_i - y_i|)^2 ) )

The classifier is built around a small set of technical‑analysis features
(RSI, CCI, ADX, EMA delta, SMA delta) that are also used by the original
TradingView indicator.

The implementation is deliberately lightweight: the model stores a
sub‑sampled library of feature vectors and performs a single‑step inference
without a training loop.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from sklearn.metrics import roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]

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
        Tensor of shape ``(..., n_features)`` representing the query vectors.
    y : torch.Tensor
        Tensor of shape ``(..., n_features)`` representing the reference vectors.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(...,)`` containing the Lorentzian distances.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for lorentzian_distance — install with `pip install torch`"
        )
    # The distance is sqrt( sum( log(1 + |x - y|)^2 ) )
    return torch.sqrt(torch.sum(torch.log(1 + torch.abs(x - y)) ** 2, dim=-1))


def compute_lorentzian_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the five technical‑analysis features required by the Lorentzian KNN.

    The function adds the following columns to a copy of ``df``:

    * ``rsi_14`` – Normalised Relative Strength Index (0‑1).
    * ``cci_20`` – Normalised Commodity Channel Index (clipped to ``[-1, 1]``).
    * ``adx_20`` – Normalised Average Directional Index (0‑1).
    * ``ema_fast_delta`` – Relative difference between fast and slow EMA.
    * ``ema_slow_delta`` – Relative difference between slow EMA and a long‑term EMA.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the ``close`` column and optionally
        ``high`` and ``low`` columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame that includes the original data plus the five new
        feature columns.
    """
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
        adx_df["ADX_20"] / 100.0
        if (adx_df is not None and "ADX_20" in adx_df.columns)
        else 0.5
    )

    ema_fast = close.ewm(span=9).mean()
    ema_slow = close.ewm(span=21).mean()
    ema_200 = close.ewm(span=200).mean()
    df["ema_fast_delta"] = (ema_fast - ema_slow) / (close + 1e-9)
    df["ema_slow_delta"] = (ema_slow - ema_200) / (close + 1e-9)

    return df


# --------------------------------------------------------------------------- #
# Model definition
# --------------------------------------------------------------------------- #

class LorentzianKNN(AbstractModel):
    """
    K‑Nearest Neighbours classifier using the Lorentzian distance metric.

    The model stores a library of historical feature vectors (``_library_X``)
    and their associated binary labels (``_library_y``). Inference consists of
    finding the ``k`` nearest neighbours of a query vector and returning the
    mean label of those neighbours.

    Parameters
    ----------
    k : int, default 8
        Number of neighbours to consider when making a prediction.
    lookback : int, default 2000
        Maximum number of historical bars to retain in the library.
    subsample : int, default 4
        Periodicity for subsampling the training data (every ``subsample``‑th
        bar is kept).
    """

    model_type = "lorentzian_knn"

    def __init__(self, k: int = 8, lookback: int = 2000, subsample: int = 4) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for LorentzianKNN — install with `pip install torch`"
            )
        self.k = k
        self.lookback = lookback
        self.subsample = subsample
        self._library_X: Optional[torch.Tensor] = None
        self._library_y: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a single‑step inference using the stored library.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, n_features)`` representing one or
            more query vectors.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch,)`` containing the predicted probabilities.
            If the library is empty, a zero tensor is returned.
        """
        if self._library_X is None:
            return torch.zeros(x.shape[0])

        results = []
        for i in range(x.shape[0]):
            query = x[i].unsqueeze(0)  # (1, n_features)
            dists = lorentzian_distance(
                query.expand_as(self._library_X), self._library_X
            )
            _, top_k = torch.topk(dists, self.k, largest=False)
            k_labels = self._library_y[top_k].float()
            results.append(k_labels.mean())
        return torch.stack(results)

    def fit_library(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Populate the KNN library from training data.

        The data is optionally subsampled and truncated to respect the
        ``lookback`` limit.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y : np.ndarray
            Binary label vector of shape ``(n_samples,)``.
        """
        idx = np.arange(0, len(X), self.subsample)
        if len(idx) > self.lookback:
            idx = idx[-self.lookback:]
        self._library_X = torch.tensor(X[idx], dtype=torch.float32)
        self._library_y = torch.tensor(y[idx], dtype=torch.float32)

    def train_epoch(
        self,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: Any = None,
        criterion: Any = None,
    ) -> Dict[str, float]:
        """
        No‑op training step for KNN.

        The Lorentzian KNN does not have learnable parameters; the library is
        built directly via ``fit_library``. This method exists to satisfy the
        ``AbstractModel`` interface.

        Returns
        -------
        dict
            A dictionary with dummy ``loss`` and ``accuracy`` values.
        """
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> EvalMetrics:
        """
        Evaluate the model on a validation set.

        Parameters
        ----------
        loader : iterable
            An iterator that yields ``(X, y)`` tuples where ``X`` is a
            ``torch.Tensor`` of shape ``(batch, n_features)`` and ``y`` is a
            ``torch.Tensor`` of shape ``(batch,)`` containing binary labels.

        Returns
        -------
        EvalMetrics
            Named tuple containing ``accuracy``, ``auc`` and ``sharpe``.
        """
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
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

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Serialize the model to disk.

        Parameters
        ----------
        path : str
            Destination file path.
        metadata : dict, optional
            Additional metadata to store alongside the model (currently unused).
        """
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
        """
        Load a previously saved ``LorentzianKNN`` instance.

        Parameters
        ----------
        path : str
            File path from which to load the model.

        Returns
        -------
        LorentzianKNN
            The deserialized model instance.
        """
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(k=data["k"], lookback=data["lookback"])
        model._library_X = data["library_X"]
        model._library_y = data["library_y"]
        return model