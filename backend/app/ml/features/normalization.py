import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """A wrapper around scikit‑learn's ``StandardScaler`` that adds convenient
    ``fit``, ``transform`` and ``fit_transform`` methods as well as persistence
    utilities for saving to and loading from disk.

    The scaler is intended for feature preprocessing in the ML pipeline; it
    stores the fitted parameters internally and can be reused for inference
    without re‑training.
    """

    def __init__(self) -> None:
        """Create a new, unfitted ``FeatureScaler`` instance."""
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """Fit the internal ``StandardScaler`` to the provided data.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The input data to compute mean and variance for scaling.

        Returns
        -------
        FeatureScaler
            The instance itself, allowing method chaining.
        """
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Scale the data using the parameters learned during ``fit``.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Data to be transformed.

        Returns
        -------
        numpy.ndarray
            The scaled representation of ``X``.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called prior to ``transform``.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit the scaler to ``X`` and then transform ``X`` in a single step.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Data to fit and transform.

        Returns
        -------
        numpy.ndarray
            The scaled representation of ``X``.
        """
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        """Persist the fitted scaler to a file using ``pickle``.

        Parameters
        ----------
        path : str
            Destination file path. Parent directories are created if they do not
            exist.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a previously saved ``FeatureScaler`` from disk.

        Parameters
        ----------
        path : str
            Path to the file containing a pickled ``StandardScaler``.

        Returns
        -------
        FeatureScaler
            A ``FeatureScaler`` instance with the loaded scaler and marked as
            fitted.
        """
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance