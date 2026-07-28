import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around scikit‑learn's ``StandardScaler`` that adds convenient
    ``fit``/``transform`` methods together with persistence utilities for
    inference.

    The scaler is stateful – after calling :meth:`fit` the instance is marked as
    fitted and can subsequently be used to transform new data.  The fitted
    scaler can be saved to disk with :meth:`save` and later restored with
    :meth:`load`, which also restores the ``fitted`` flag.
    """

    def __init__(self) -> None:
        """Create a new, unfitted ``FeatureScaler``."""
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """Fit the internal ``StandardScaler`` to the provided data.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Training data used to compute the mean and variance for each feature.

        Returns
        -------
        FeatureScaler
            The instance itself, allowing method chaining.
        """
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using the previously fitted scaler.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Data to be standardized. Must have the same number of features as the
            data used in :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            The standardized data.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit the scaler to ``X`` and then transform ``X`` in a single step.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Data used for fitting and transformation.

        Returns
        -------
        numpy.ndarray
            The standardized data.
        """
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        """Persist the fitted scaler to a file.

        The target directory is created automatically if it does not exist.

        Parameters
        ----------
        path : str
            File path where the scaler will be serialized using ``pickle``.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a previously saved scaler from disk.

        Parameters
        ----------
        path : str
            File path from which the scaler will be deserialized.

        Returns
        -------
        FeatureScaler
            An instance with the loaded ``StandardScaler`` and ``fitted`` flag set.
        """
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance