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

    def _log_metric(self, action: str, count: int = 0, duration: float = 0.0, pnl: float | None = None) -> None:
        """
        Log structured metrics at INFO level.

        Parameters
        ----------
        action : str
            Name of the operation (e.g., 'fit', 'transform').
        count : int
            Number of signals/rows processed.
        duration : float
            Execution time in seconds.
        pnl : float | None
            Profit & loss associated with the operation, if applicable.
        """
        log_record = {
            "action": action,
            "signal_count": count,
            "execution_time_ms": round(duration * 1000, 3),
        }
        if pnl is not None:
            log_record["pnl"] = pnl
        logger.info(log_record)

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        start = time.perf_counter()
        self.scaler.fit(X)
        self.fitted = True
        duration = time.perf_counter() - start
        count = X.shape[0] if hasattr(X, "shape") else 0
        self._log_metric(action="fit", count=count, duration=duration)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        start = time.perf_counter()
        result = self.scaler.transform(X)
        duration = time.perf_counter() - start
        count = X.shape[0] if hasattr(X, "shape") else 0
        self._log_metric(action="transform", count=count, duration=duration)
        return result

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        result = self.fit(X).transform(X)
        duration = time.perf_counter() - start
        count = X.shape[0] if hasattr(X, "shape") else 0
        self._log_metric(action="fit_transform", count=count, duration=duration)
        return result

    def save(self, path: str) -> None:
        start = time.perf_counter()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        duration = time.perf_counter() - start
        self._log_metric(action="save", count=0, duration=duration)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        start = time.perf_counter()
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        duration = time.perf_counter() - start
        logger.info(
            {
                "action": "load",
                "signal_count": 0,
                "execution_time_ms": round(duration * 1000, 3),
            }
        )
        return instance