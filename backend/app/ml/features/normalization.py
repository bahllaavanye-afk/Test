import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around scikit-learn's ``StandardScaler`` that adds convenient
    save/load functionality for use during inference.

    The scaler can be fitted on a pandas ``DataFrame`` or a NumPy array and
    later reused to transform new data. The fitted scaler can be persisted to
    disk with :meth:`save` and re‑loaded with :meth:`load`.
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
            Training data used to compute the mean and variance for scaling.

        Returns
        -------
        FeatureScaler
            The same instance, allowing method chaining.
        """
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Scale ``X`` using the parameters learned during :meth:`fit`.

        Parameters
        ----------
        X: pandas.DataFrame or numpy.ndarray
            Data to be scaled.

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
        """Fit the scaler to ``X`` and return the transformed data.

        This is a convenience method equivalent to ``self.fit(X).transform(X)``.

        Parameters
        ----------
        X: pandas.DataFrame or numpy.ndarray
            Data to fit and transform.

        Returns
        -------
        numpy.ndarray
            The scaled version of ``X``.
        """
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted scaler to disk.

        Parameters
        ----------
        path: str or pathlib.Path
            Destination file path where the scaler will be serialized.
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
            File path from which to deserialize the scaler.

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