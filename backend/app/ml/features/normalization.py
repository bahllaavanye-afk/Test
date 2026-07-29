import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def _validate_input(self, X):
        """Validate that X is a non‑empty DataFrame or ndarray."""
        if X is None:
            raise ValueError("Input data X cannot be None")
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input DataFrame X is empty")
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Input ndarray X is empty")
        else:
            raise TypeError(
                f"Input X must be a pandas DataFrame or numpy ndarray, got {type(X)}"
            )
        return X

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        X = self._validate_input(X)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X = self._validate_input(X)
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = self._validate_input(X)
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        if not path:
            raise ValueError("Save path must be a non‑empty string")
        # Ensure the directory exists; pathlib handles off‑by‑one slash issues
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        if not path:
            raise ValueError("Load path must be a non‑empty string")
        instance = cls()
        try:
            with open(path, "rb") as f:
                instance.scaler = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Scaler file not found at '{path}'") from e
        instance.fitted = True
        return instance