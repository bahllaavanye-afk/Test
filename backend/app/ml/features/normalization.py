import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around ``StandardScaler`` with save/load capabilities for inference.

    The class tracks whether the scaler has been fitted and provides a simple
    interface for fitting, transforming, and persisting the scaler.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted: bool = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """Fit the internal ``StandardScaler`` to ``X``."""
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform ``X`` using the fitted scaler.

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
        """Persist the fitted scaler to ``path``.

        The directory hierarchy is created if it does not exist. The
        operation is atomic to avoid partial writes.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with open(temp_path, "wb") as f:
            pickle.dump(self.scaler, f)
        temp_path.replace(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureScaler":
        """Load a previously saved scaler from ``path``.

        Parameters
        ----------
        path : str or Path
            Location of the saved scaler file.

        Returns
        -------
        FeatureScaler
            An instance with the loaded scaler and ``fitted`` flag set to ``True``.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No scaler file found at {path}")

        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance


__all__ = ["FeatureScaler"]