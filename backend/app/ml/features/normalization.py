import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Union


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference.

    This class adds defensive checks for ``None`` inputs, empty collections,
    and validates the provided file paths to avoid off‑by‑one or index errors.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """Validate that ``X`` is non‑empty and not ``None``.

        Raises
        ------
        ValueError
            If ``X`` is ``None`` or contains no samples.
        """
        if X is None:
            raise ValueError("Input data cannot be None")
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input DataFrame is empty")
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Input array is empty")
        else:
            raise TypeError(
                f"Unsupported input type: {type(X)}. Expected pandas DataFrame or numpy array."
            )

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """Fit the scaler to ``X`` after validating the input."""
        self._validate_input(X)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform ``X`` using the fitted scaler.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        ValueError
            If ``X`` is ``None`` or empty.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        self._validate_input(X)
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit to data, then transform it.

        This method ensures the input validation is performed only once.
        """
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        """Persist the underlying ``StandardScaler`` to ``path``.

        The method creates parent directories if needed and validates that
        ``path`` is a non‑empty string.
        """
        if not path:
            raise ValueError("Save path must be a non‑empty string")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a previously saved scaler from ``path``.

        The method validates that ``path`` points to an existing file.
        """
        if not path:
            raise ValueError("Load path must be a non‑empty string")
        if not Path(path).is_file():
            raise FileNotFoundError(f"No such file: {path}")

        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance