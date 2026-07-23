import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Utility wrapper around :class:`sklearn.preprocessing.StandardScaler`.

    Provides a simple interface for fitting, transforming, and persisting a
    scaler instance. The class tracks whether the scaler has been fitted to
    guard against accidental usage before training.
    """

    def __init__(self) -> None:
        """Initialize an unfitted ``StandardScaler``."""
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """Fit the underlying scaler to the data.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Training data used to compute the mean and variance.

        Returns
        -------
        FeatureScaler
            The instance itself, allowing method chaining.
        """
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using the fitted scaler.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Data to be scaled.

        Returns
        -------
        numpy.ndarray
            Scaled representation of ``X``.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit the scaler to ``X`` and then transform ``X``.

        This convenience method combines :meth:`fit` and :meth:`transform`.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Data to fit and transform.

        Returns
        -------
        numpy.ndarray
            Scaled version of ``X`` after fitting.
        """
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted scaler to disk using ``pickle``.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path. Parent directories are created if they do not exist.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureScaler":
        """Load a previously saved scaler from disk.

        Parameters
        ----------
        path : str or pathlib.Path
            File path from which to load the scaler.

        Returns
        -------
        FeatureScaler
            An instance with the scaler restored and marked as fitted.
        """
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance