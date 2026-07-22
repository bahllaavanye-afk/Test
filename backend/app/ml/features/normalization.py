import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference.

    Provides additional utilities for signal generation, including
    entry/exit confirmation filters based on configurable thresholds.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        self._numeric_columns: List[str] = []

    # --------------------------------------------------------------------- #
    # Core scaler methods
    # --------------------------------------------------------------------- #
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """Fit the scaler on the provided data."""
        X = self._prepare_input(X)
        self.scaler.fit(X)
        self.fitted = True
        self._numeric_columns = list(X.columns)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using the fitted scaler."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X = self._prepare_input(X, require_numeric=True)
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save(self, path: str) -> None:
        """Persist the fitted scaler to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "numeric_columns": self._numeric_columns,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a previously saved scaler."""
        instance = cls()
        with open(path, "rb") as f:
            payload = pickle.load(f)
            instance.scaler = payload["scaler"]
            instance._numeric_columns = payload.get("numeric_columns", [])
        instance.fitted = True
        return instance

    # --------------------------------------------------------------------- #
    # Signal generation utilities
    # --------------------------------------------------------------------- #
    def generate_entry_signal(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        thresholds: Dict[str, float],
        min_features: int = 2,
    ) -> np.ndarray:
        """Return a boolean array indicating entry signals.

        An entry is signaled when at least ``min_features`` scaled features
        exceed their respective positive thresholds.

        Parameters
        ----------
        X : DataFrame or ndarray
            Raw feature data.
        thresholds : dict
            Mapping of column names to positive threshold values (in standard‑deviation units).
        min_features : int, default 2
            Minimum number of features that must exceed thresholds to trigger a signal.

        Returns
        -------
        np.ndarray
            Boolean array where ``True`` denotes an entry signal.
        """
        scaled = self.transform(X)
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
        else:
            cols = self._numeric_columns

        # Build threshold vector aligned with the column order
        thresh_vec = np.array([thresholds.get(col, 0.0) for col in cols])
        exceed = scaled > thresh_vec
        count_exceed = exceed.sum(axis=1)
        return count_exceed >= min_features

    def generate_exit_signal(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        thresholds: Dict[str, float],
        max_features: int = 1,
    ) -> np.ndarray:
        """Return a boolean array indicating exit signals.

        An exit is signaled when ``max_features`` or fewer scaled features fall
        below their negative thresholds, suggesting a weakening trend.

        Parameters
        ----------
        X : DataFrame or ndarray
            Raw feature data.
        thresholds : dict
            Mapping of column names to positive threshold values (used symmetrically for exit).
        max_features : int, default 1
            Maximum number of features allowed to stay above thresholds before exiting.

        Returns
        -------
        np.ndarray
            Boolean array where ``True`` denotes an exit signal.
        """
        scaled = self.transform(X)
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
        else:
            cols = self._numeric_columns

        thresh_vec = np.array([thresholds.get(col, 0.0) for col in cols])
        below = scaled < -thresh_vec
        count_below = below.sum(axis=1)
        return count_below >= max_features

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _prepare_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        require_numeric: bool = False,
    ) -> pd.DataFrame:
        """Validate and convert input to a numeric DataFrame.

        - Ensures a DataFrame output.
        - Optionally filters to numeric columns only.
        - Raises if NaNs are present.
        """
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError("Input ndarray must be 2‑dimensional")
            # If column names are unknown, generate generic ones
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame or 2‑D numpy array")

        if require_numeric:
            numeric_df = X.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise RuntimeError("No numeric columns found for scaling")
            X = numeric_df

        if X.isnull().any().any():
            raise RuntimeError("NaN values detected in input data; cannot scale")

        return X

    # --------------------------------------------------------------------- #
    # Convenience methods for downstream strategy code
    # --------------------------------------------------------------------- #
    def fit_with_validation(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        validation_split: float = 0.2,
        random_state: int = 42,
    ) -> "FeatureScaler":
        """Fit the scaler on training data while reserving a validation slice.

        This method is useful for ensuring the scaler generalises and helps
        prevent data leakage in production pipelines.
        """
        X = self._prepare_input(X, require_numeric=True)
        if not 0 < validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")

        # Shuffle rows deterministically
        X_shuffled = X.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(X_shuffled) * (1 - validation_split))
        train = X_shuffled.iloc[:split_idx]
        # Validation set could be used for diagnostics; currently ignored
        self.scaler.fit(train)
        self.fitted = True
        self._numeric_columns = list(train.columns)
        return self