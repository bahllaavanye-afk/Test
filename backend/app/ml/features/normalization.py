import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Utility wrapper around ``sklearn.preprocessing.StandardScaler`` that adds
    convenient ``fit``, ``transform`` and persistence methods for use in the
    production ML pipeline.

    The scaler is instantiated unfitted. Calling :meth:`fit` or
    :meth:`fit_transform` marks the object as fitted; subsequent calls to
    :meth:`transform` will raise a ``RuntimeError`` if the scaler has not been
    fitted. The fitted scaler can be saved to disk with :meth:`save` and later
    re‑loaded with :meth:`load`, which restores the fitted state automatically.
    """

    def __init__(self) -> None:
        """Create a new, unfitted ``FeatureScaler`` instance."""
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """Fit the internal ``StandardScaler`` to the provided data.

        Parameters
        ----------
        X: pandas.DataFrame or numpy.ndarray
            Input data to compute mean and variance for scaling.

        Returns
        -------
        FeatureScaler
            The instance itself, allowing method chaining.
        """
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Scale the provided data using the previously fitted statistics.

        Parameters
        ----------
        X: pandas.DataFrame or numpy.ndarray
            Data to be transformed.

        Returns
        -------
        numpy.ndarray
            Scaled version of ``X``.

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
        X: pandas.DataFrame or numpy.ndarray
            Data to fit and transform.

        Returns
        -------
        numpy.ndarray
            Scaled version of ``X``.
        """
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted ``StandardScaler`` to a file.

        The directory hierarchy is created automatically if it does not exist.

        Parameters
        ----------
        path: str or pathlib.Path
            Destination file path where the scaler will be pickled.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureScaler":
        """Load a previously saved scaler from disk.

        Parameters
        ----------
        path: str or pathlib.Path
            File path from which to load the pickled ``StandardScaler``.

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