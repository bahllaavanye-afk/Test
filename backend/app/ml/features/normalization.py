import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ArrayLike = Union[pd.DataFrame, np.ndarray]


class FeatureScaler:
    """A thin wrapper around scikit‑learn's :class:`StandardScaler` that adds
    persistence, validation and a small set of utilities useful for trading
    feature pipelines.

    The scaler is *stateful*: it must be fitted on training data before it can be
    used to transform live data.  The :pyattr:`fitted` flag tracks this state.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    # --------------------------------------------------------------------- #
    # Core API – fit / transform / fit_transform
    # --------------------------------------------------------------------- #
    def fit(self, X: ArrayLike) -> "FeatureScaler":
        """Fit the scaler to ``X`` after validating the input.

        Parameters
        ----------
        X: pandas.DataFrame or numpy.ndarray
            Training data. Must be two‑dimensional and contain no NaNs.

        Returns
        -------
        FeatureScaler
            Self, to allow method chaining.
        """
        X = self._validate_input(X, require_finite=True)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform ``X`` using the fitted scaler.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X = self._validate_input(X, require_finite=True)
        return self.scaler.transform(X)

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Fit to ``X`` then transform it in a single step."""
        return self.fit(X).transform(X)

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #
    def save(self, path: str) -> None:
        """Serialise the underlying ``StandardScaler`` to ``path``."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a previously saved scaler from ``path``."""
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance

    # --------------------------------------------------------------------- #
    # Additional utilities
    # --------------------------------------------------------------------- #
    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Revert a transformed array back to the original scale.

        Parameters
        ----------
        X: pandas.DataFrame or numpy.ndarray
            Data that was previously transformed with this scaler.

        Returns
        -------
        numpy.ndarray
            The data in the original feature space.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X = self._validate_input(X, require_finite=True)
        return self.scaler.inverse_transform(X)

    def get_params(self) -> dict:
        """Return the mean and scale learned during fitting.

        Useful for logging or debugging feature pipelines.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return {"mean": self.scaler.mean_, "scale": self.scaler.scale_}

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _validate_input(
        X: ArrayLike, *, require_finite: bool = False
    ) -> np.ndarray:
        """Validate that ``X`` is a 2‑D array with optional finiteness checks.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Input data.
        require_finite : bool, default False
            If True, NaN/inf values raise a ``ValueError``.

        Returns
        -------
        numpy.ndarray
            The validated data as a NumPy array.
        """
        if isinstance(X, pd.DataFrame):
            arr = X.values
        elif isinstance(X, np.ndarray):
            arr = X
        else:
            raise TypeError(
                f"Unsupported input type {type(X)}; expected DataFrame or ndarray."
            )

        if arr.ndim != 2:
            raise ValueError(
                f"Input must be 2‑dimensional, got shape {arr.shape}."
            )

        if require_finite and not np.isfinite(arr).all():
            raise ValueError("Input contains NaN or infinite values.")

        return arr

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "unfitted"
        return f"{self.__class__.__name__}({status})"