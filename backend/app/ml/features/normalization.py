import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference.

    Provides additional utilities for signal generation and data validation
    to help tighten entry conditions, add confirmation filters, and improve
    exit logic in trading strategies.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        """Fit the scaler to the provided data."""
        X = self._validate_input(X)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler.

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X = self._validate_input(X)
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        X = self._validate_input(X)
        return self.scaler.fit_transform(X)

    def inverse_transform(self, X_scaled: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Revert scaled data back to original space."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — cannot inverse_transform")
        X_scaled = self._validate_input(X_scaled)
        return self.scaler.inverse_transform(X_scaled)

    def save(self, path: str) -> None:
        """Persist the fitted scaler to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a previously saved scaler."""
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance

    # --------------------------------------------------------------------- #
    # Extended utilities for strategy signal generation
    # --------------------------------------------------------------------- #

    def generate_signal(
        self,
        X: pd.DataFrame | np.ndarray,
        entry_z: float = 1.0,
        exit_z: float = 0.5,
        consecutive: int = 2,
    ) -> pd.Series:
        """Generate binary entry/exit signals based on scaled feature values.

        The method computes a row‑wise average of the scaled features and
        applies the following logic:

        * **Entry condition** – the average exceeds ``entry_z`` for at
          least ``consecutive`` consecutive rows.
        * **Exit condition** – after an entry, the average falls below
          ``exit_z`` for at least ``consecutive`` consecutive rows.

        Parameters
        ----------
        X : DataFrame or ndarray
            Raw feature matrix.
        entry_z : float, default 1.0
            Z‑score threshold to trigger a potential entry.
        exit_z : float, default 0.5
            Z‑score threshold to trigger an exit.
        consecutive : int, default 2
            Number of consecutive periods required to confirm entry or exit.

        Returns
        -------
        pd.Series
            Signal series indexed like ``X`` with values:
            * ``1`` – confirmed entry,
            * ``0`` – confirmed exit,
            * ``np.nan`` – no signal / waiting for confirmation.
        """
        X = self._validate_input(X)
        # Ensure scaler is fitted; if not, fit on‑the‑fly (safe fallback)
        if not self.fitted:
            self.fit(X)

        scaled = pd.DataFrame(self.transform(X), index=self._extract_index(X))
        # Use row‑wise mean as a simple composite indicator
        composite = scaled.mean(axis=1)

        # Rolling windows for confirmation
        entry_cond = composite.rolling(consecutive).apply(
            lambda w: np.all(w > entry_z), raw=True
        )
        exit_cond = composite.rolling(consecutive).apply(
            lambda w: np.all(w < exit_z), raw=True
        )

        signals = pd.Series(np.nan, index=composite.index, dtype=float)

        # Identify entry points
        entry_idxs = entry_cond[entry_cond == 1.0].index
        signals.loc[entry_idxs] = 1.0

        # Identify exit points after an entry
        if not entry_idxs.empty:
            # Propagate entry state forward until exit condition met
            in_position = False
            for idx in composite.index:
                if idx in entry_idxs:
                    in_position = True
                    continue
                if in_position and idx in exit_cond[exit_cond == 1.0].index:
                    signals.loc[idx] = 0.0
                    in_position = False

        return signals

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _validate_input(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Validate and convert input to a DataFrame.

        - Ensures numeric dtype.
        - Removes or raises on NaNs.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame or a numpy ndarray")

        if not np.issubdtype(X.dtypes.values, np.number):
            raise ValueError("All columns must be numeric for scaling")

        if X.isnull().any().any():
            raise ValueError("Input contains NaN values; clean data before scaling")
        return X

    @staticmethod
    def _extract_index(X: pd.DataFrame | np.ndarray) -> Iterable:
        """Return the index to be used for the transformed DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.index
        # For ndarray fallback to default RangeIndex
        return range(X.shape[0])