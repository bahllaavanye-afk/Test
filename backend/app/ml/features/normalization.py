import pickle
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference.

    Adds structured INFO‑level logging for key metrics:
    * ``signal_count`` – number of rows processed.
    * ``execution_time`` – duration of the operation in seconds.
    * ``pnl`` – placeholder for profit & loss (if applicable).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def _log(self, operation: str, count: int, elapsed: float, pnl: float | None = None) -> None:
        """Emit a structured log message for the given operation."""
        log_payload = {
            "operation": operation,
            "signal_count": count,
            "execution_time": elapsed,
            "pnl": pnl,
        }
        logger.info("%s %s", operation, log_payload)

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        start = time.perf_counter()
        self.scaler.fit(X)
        self.fitted = True
        elapsed = time.perf_counter() - start
        count = X.shape[0] if hasattr(X, "shape") else 0
        self._log("fit", count, elapsed)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        start = time.perf_counter()
        result = self.scaler.transform(X)
        elapsed = time.perf_counter() - start
        count = X.shape[0] if hasattr(X, "shape") else 0
        self._log("transform", count, elapsed)
        return result

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        result = self.fit(X).transform(X)
        elapsed = time.perf_counter() - start
        count = X.shape[0] if hasattr(X, "shape") else 0
        self._log("fit_transform", count, elapsed)
        return result

    def save(self, path: str) -> None:
        start = time.perf_counter()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        elapsed = time.perf_counter() - start
        # Saving does not involve a data matrix; signal count is set to 0.
        self._log("save", 0, elapsed)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        start = time.perf_counter()
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        elapsed = time.perf_counter() - start
        # Loading does not involve a data matrix; signal count is set to 0.
        logger.info(
            "load %s",
            {
                "operation": "load",
                "signal_count": 0,
                "execution_time": elapsed,
                "pnl": None,
            },
        )
        return instance