import pickle
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _extract_signal_count(X):
    """Return the number of rows (signals) in the input."""
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return int(X.shape[0])
    if isinstance(X, np.ndarray):
        return int(X.shape[0])
    return 0


def _extract_pnl(X):
    """Attempt to compute total P&L if a 'pnl' column exists."""
    if isinstance(X, pd.DataFrame) and "pnl" in X.columns:
        try:
            return float(X["pnl"].sum())
        except Exception:
            pass
    return None


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        start_time = time.time()
        self.scaler.fit(X)
        self.fitted = True
        duration_ms = (time.time() - start_time) * 1000
        signal_count = _extract_signal_count(X)
        pnl = _extract_pnl(X)
        logger.info(
            {
                "event": "fit",
                "signal_count": signal_count,
                "duration_ms": round(duration_ms, 2),
                "pnl": pnl,
                "status": "completed",
            }
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        start_time = time.time()
        result = self.scaler.transform(X)
        duration_ms = (time.time() - start_time) * 1000
        signal_count = _extract_signal_count(X)
        pnl = _extract_pnl(X)
        logger.info(
            {
                "event": "transform",
                "signal_count": signal_count,
                "duration_ms": round(duration_ms, 2),
                "pnl": pnl,
                "status": "completed",
            }
        )
        return result

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        start_time = time.time()
        result = self.fit(X).transform(X)
        duration_ms = (time.time() - start_time) * 1000
        signal_count = _extract_signal_count(X)
        pnl = _extract_pnl(X)
        logger.info(
            {
                "event": "fit_transform",
                "signal_count": signal_count,
                "duration_ms": round(duration_ms, 2),
                "pnl": pnl,
                "status": "completed",
            }
        )
        return result

    def save(self, path: str) -> None:
        start_time = time.time()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            {
                "event": "save",
                "path": path,
                "duration_ms": round(duration_ms, 2),
                "status": "completed",
            }
        )

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        start_time = time.time()
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            {
                "event": "load",
                "path": path,
                "duration_ms": round(duration_ms, 2),
                "status": "completed",
            }
        )
        return instance