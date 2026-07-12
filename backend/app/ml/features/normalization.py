import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference.

    Handles edge cases such as None inputs, empty collections, and ensures
    inputs are two‑dimensional before delegating to ``StandardScaler``.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    @staticmethod
    def _validate_input(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Validate and normalise input for scaling.

        - Raises ``ValueError`` if ``X`` is ``None`` or empty.
        - Converts 1‑D inputs to a 2‑D column vector.
        - Returns a NumPy ``ndarray`` suitable for ``StandardScaler``.
        """
        if X is None:
            raise ValueError("Input data cannot be None")

        # Convert pandas DataFrame to numpy array early for uniform handling
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not isinstance(X, np.ndarray):
            raise TypeError(
                f"Expected input type pandas.DataFrame or numpy.ndarray, got {type(X)}"
            )

        if X.size == 0:
            raise ValueError("Input data is empty")

        # Ensure two‑dimensional array; StandardScaler expects shape (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        X = self._validate_input(X)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X = self._validate_input(X)
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = self._validate_input(X)
        self.scaler.fit(X)
        self.fitted = True
        return self.scaler.transform(X)

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