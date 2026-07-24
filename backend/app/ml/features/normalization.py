import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around ``StandardScaler`` with persistence utilities and basic signal generation.

    The class provides a thin layer over ``StandardScaler`` that adds:
    * explicit fit state tracking,
    * input validation for both pandas and numpy inputs,
    * convenience methods for saving/loading the scaler,
    * a lightweight signal generation helper that can be used by trading strategies
      to tighten entry/exit conditions based on normalized features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    # --------------------------------------------------------------------- #
    # Core scaler API
    # --------------------------------------------------------------------- #
    def _validate_input(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Validate that *X* is a 2‑dimensional array-like suitable for the scaler.

        Raises:
            ValueError: If *X* is not 2‑dimensional or contains NaNs.
        """
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input DataFrame is empty.")
            if X.isnull().any().any():
                raise ValueError("Input DataFrame contains NaN values.")
        elif isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    f"Numpy input must be 2‑dimensional, got shape {X.shape}."
                )
            if np.isnan(X).any():
                raise ValueError("Input numpy array contains NaN values.")
        else:
            raise TypeError(
                "Input must be a pandas DataFrame or a 2‑dimensional numpy array."
            )
        return X

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """Fit the internal ``StandardScaler`` on *X*."""
        X = self._validate_input(X)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform *X* using the fitted scaler.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X = self._validate_input(X)
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit the scaler on *X* and return the transformed values."""
        return self.fit(X).transform(X)

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #
    def save(self, path: str) -> None:
        """Serialise the underlying ``StandardScaler`` to *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a previously saved scaler from *path*."""
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance

    # --------------------------------------------------------------------- #
    # Strategy‑focused utilities
    # --------------------------------------------------------------------- #
    def generate_signals(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_col: Union[str, int],
        entry_threshold: float = 0.0,
        exit_threshold: float = 0.0,
        confirmation_cols: Optional[Iterable[Union[str, int]]] = None,
        confirmation_window: int = 3,
        confirmation_slope: float = 0.0,
    ) -> pd.DataFrame:
        """Generate entry/exit signals based on a normalized feature.

        Parameters
        ----------
        X : DataFrame or ndarray
            Raw (pre‑scaling) feature matrix. It will be validated and transformed
            using the internal scaler.
        feature_col : str or int
            Column name (if ``X`` is a DataFrame) or column index (if ``X`` is an
            ndarray) that represents the primary signal generator.
        entry_threshold : float, default 0.0
            Normalized value above which an entry signal (``1``) is considered.
        exit_threshold : float, default 0.0
            Normalized value below which an exit signal (``-1``) is considered.
        confirmation_cols : iterable of str or int, optional
            Additional columns that must show a consistent upward (or downward)
            trend over ``confirmation_window`` periods to confirm the entry (or
            exit) decision.
        confirmation_window : int, default 3
            Number of periods used for the rolling confirmation check.
        confirmation_slope : float, default 0.0
            Minimum required slope (difference between last and first value in the
            window) for a confirmation column to be deemed supportive.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed like the input with a single column ``signal``:
            * ``1`` – entry,
            * ``-1`` – exit,
            * ``0`` – no action.
        """
        # Validate and scale the raw input
        X_valid = self._validate_input(X)
        X_scaled = self.transform(X_valid)

        # Convert to DataFrame for easier column handling
        if isinstance(X_valid, pd.DataFrame):
            cols = list(X_valid.columns)
            df_scaled = pd.DataFrame(
                X_scaled, index=X_valid.index, columns=cols
            )
        else:
            # ndarray case – generate generic column names
            n_cols = X_scaled.shape[1]
            cols = [f"col_{i}" for i in range(n_cols)]
            df_scaled = pd.DataFrame(
                X_scaled, index=pd.RangeIndex(len(X_scaled)), columns=cols
            )

        # Primary feature series
        primary_series = (
            df_scaled[feature_col]
            if isinstance(feature_col, str)
            else df_scaled.iloc[:, feature_col]
        )

        # Initialise signal series with zeros
        signals = pd.Series(0, index=primary_series.index, dtype=int)

        # Entry condition: primary > entry_threshold
        entry_mask = primary_series > entry_threshold

        # Exit condition: primary < exit_threshold
        exit_mask = primary_series < exit_threshold

        # Apply confirmation filters if requested
        if confirmation_cols:
            # Build a mask that is True only when *all* confirmation columns satisfy
            # the required slope over the rolling window.
            confirmation_mask = pd.Series(True, index=primary_series.index)

            for col in confirmation_cols:
                series = (
                    df_scaled[col]
                    if isinstance(col, str)
                    else df_scaled.iloc[:, col]
                )
                # Compute rolling slope (last - first in the window)
                rolling_slope = series.diff(periods=confirmation_window - 1)
                # Positive slope for entry, negative for exit (handled later)
                col_mask = rolling_slope >= confirmation_slope
                confirmation_mask &= col_mask

            entry_mask &= confirmation_mask
            # For exits we require a downward trend; reuse the same mask with
            # opposite slope.
            exit_confirmation_mask = pd.Series(True, index=primary_series.index)
            for col in confirmation_cols:
                series = (
                    df_scaled[col]
                    if isinstance(col, str)
                    else df_scaled.iloc[:, col]
                )
                rolling_slope = series.diff(periods=confirmation_window - 1)
                col_mask = rolling_slope <= -confirmation_slope
                exit_confirmation_mask &= col_mask

            exit_mask &= exit_confirmation_mask

        # Assign signals
        signals[entry_mask] = 1
        signals[exit_mask] = -1

        return pd.DataFrame({"signal": signals}, index=signals.index)