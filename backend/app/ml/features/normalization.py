import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


"""Normalization utilities for feature scaling.

This module provides a thin wrapper around scikit‑learn's ``StandardScaler`` that
adds convenient ``save`` and ``load`` methods for persisting the scaler state
between training and inference runs.
"""


class FeatureScaler:
    """Wrapper around :class:`sklearn.preprocessing.StandardScaler` with persistence support.

    The class tracks whether the scaler has been fitted and raises a clear error
    if ``transform`` is called before fitting.  It can be saved to and loaded from
    disk using ``pickle``.
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
            Training data used to compute the mean and standard deviation for each
            feature. The shape should be ``(n_samples, n_features)``.

        Returns
        -------
        FeatureScaler
            The instance itself, allowing method chaining.
        """
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Scale the data using the fitted parameters.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Data to be transformed. Must have the same number of features as the
            data used in ``fit``.

        Returns
        -------
        numpy.ndarray
            The scaled representation of ``X``.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit to data, then transform it.

        This is a convenience method that combines ``fit`` and ``transform``.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Training data to fit and transform.

        Returns
        -------
        numpy.ndarray
            The scaled representation of ``X``.
        """
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the scaler to a file.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path where the scaler will be pickled. Parent
            directories are created automatically if they do not exist.
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
            Path to the pickled ``StandardScaler`` file.

        Returns
        -------
        FeatureScaler
            An instance with the loaded scaler and ``fitted`` flag set to ``True``.
        """
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance