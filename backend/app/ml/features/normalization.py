import pickle
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference.

    Adds structured INFO logging for key metrics:
    - signal count (number of rows processed)
    - execution time (seconds)
    - pnl (placeholder, set to ``None`` as scaling does not affect P&L)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        start_time = time.time()
        self.scaler.fit(X)
        self.fitted = True
        elapsed = time.time() - start_time

        signal_count = X.shape[0] if hasattr(X, "shape") else len(X)
        logger.info(
            "FeatureScaler fit completed",
            extra={
                "signal_count": signal_count,
                "execution_time_sec": elapsed,
                "pnl": None,
            },
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")

        start_time = time.time()
        result = self.scaler.transform(X)
        elapsed = time.time() - start_time

        signal_count = X.shape[0] if hasattr(X, "shape") else len(X)
        logger.info(
            "FeatureScaler transform completed",
            extra={
                "signal_count": signal_count,
                "execution_time_sec": elapsed,
                "pnl": None,
            },
        )
        return result

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        start_time = time.time()
        result = self.fit(X).transform(X)
        elapsed = time.time() - start_time

        signal_count = X.shape[0] if hasattr(X, "shape") else len(X)
        logger.info(
            "FeatureScaler fit_transform completed",
            extra={
                "signal_count": signal_count,
                "execution_time_sec": elapsed,
                "pnl": None,
            },
        )
        return result

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance