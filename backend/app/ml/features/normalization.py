import pickle
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        start = time.perf_counter()
        self.scaler.fit(X)
        self.fitted = True
        duration = time.perf_counter() - start
        signal_count = X.shape[0] if hasattr(X, "shape") else None
        logger.info(
            "FeatureScaler fit completed",
            extra={
                "signal_count": signal_count,
                "execution_time_sec": duration,
                "pnl": None,
            },
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        start = time.perf_counter()
        result = self.scaler.transform(X)
        duration = time.perf_counter() - start
        signal_count = X.shape[0] if hasattr(X, "shape") else None
        logger.info(
            "FeatureScaler transform completed",
            extra={
                "signal_count": signal_count,
                "execution_time_sec": duration,
                "pnl": None,
            },
        )
        return result

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        result = self.fit(X).transform(X)
        duration = time.perf_counter() - start
        signal_count = X.shape[0] if hasattr(X, "shape") else None
        logger.info(
            "FeatureScaler fit_transform completed",
            extra={
                "signal_count": signal_count,
                "execution_time_sec": duration,
                "pnl": None,
            },
        )
        return result

    def save(self, path: str) -> None:
        start = time.perf_counter()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        duration = time.perf_counter() - start
        logger.info(
            "FeatureScaler saved to disk",
            extra={
                "path": path,
                "execution_time_sec": duration,
                "signal_count": None,
                "pnl": None,
            },
        )

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        start = time.perf_counter()
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        duration = time.perf_counter() - start
        logger.info(
            "FeatureScaler loaded from disk",
            extra={
                "path": path,
                "execution_time_sec": duration,
                "signal_count": None,
                "pnl": None,
            },
        )
        return instance