import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference.

    Handles edge cases such as None inputs, empty collections, and validates
    that the scaler has been fitted before transformation.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    @staticmethod
    def _validate_input(X, context: str) -> None:
        """Validate that X is not None and contains data.

        Args:
            X: Input data (DataFrame or ndarray).
            context: Description of the operation for error messages.

        Raises:
            ValueError: If X is None or empty.
        """
        if X is None:
            raise ValueError(f"{context}: input data is None")
        if isinstance(X, (pd.DataFrame, pd.Series)):
            if X.empty:
                raise ValueError(f"{context}: input DataFrame/Series is empty")
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError(f"{context}: input ndarray is empty")
        else:
            raise TypeError(
                f"{context}: input must be a pandas DataFrame/Series or numpy ndarray"
            )

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        """Fit the scaler to the data.

        Args:
            X: Training data.

        Returns:
            self
        """
        self._validate_input(X, "fit")
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler.

        Args:
            X: Data to transform.

        Returns:
            Transformed data as a NumPy array.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        self._validate_input(X, "transform")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X: Data to fit and transform.

        Returns:
            Transformed data as a NumPy array.
        """
        self._validate_input(X, "fit_transform")
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        """Persist the fitted scaler to disk.

        Args:
            path: Destination file path.
        """
        if not path:
            raise ValueError("save: path must be a non‑empty string")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a persisted scaler from disk.

        Args:
            path: File path to load the scaler from.

        Returns:
            An instance of FeatureScaler with the loaded scaler.
        """
        if not path:
            raise ValueError("load: path must be a non‑empty string")
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance