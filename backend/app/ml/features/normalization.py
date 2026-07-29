import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around ``StandardScaler`` with convenient save/load methods for inference."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """Fit the scaler to the data."""
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using the fitted scaler.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted scaler to ``path`` using pickle.

        The parent directory is created if it does not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureScaler":
        """Load a previously saved scaler from ``path``.

        Parameters
        ----------
        path : str or Path
            Location of the pickle file.

        Returns
        -------
        FeatureScaler
            Instance with the loaded scaler marked as fitted.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No scaler file found at {path}")
        instance = cls()
        with path.open("rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance