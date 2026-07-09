import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around ``StandardScaler`` with persistence and basic validation.

    The class is used throughout the platform to normalise feature vectors before
    they are consumed by trading strategies.  In addition to the original
    ``fit``/``transform`` API, helper methods are provided to generate simple
    entry/exit signals based on the scaled data and to apply a confirmation
    filter that reduces false positives.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    # --------------------------------------------------------------------- #
    # Core scaling API
    # --------------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        """Fit the scaler to the provided data after basic validation.

        Parameters
        ----------
        X: pd.DataFrame | np.ndarray
            Input data to compute mean and variance.  NaN values are not
            permitted because they would corrupt the scaling parameters.

        Returns
        -------
        FeatureScaler
            The instance itself, allowing method chaining.
        """
        X_clean = self._validate_input(X)
        self.scaler.fit(X_clean)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Scale the input using the previously fitted parameters.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X_clean = self._validate_input(X, allow_nan=False)
        return self.scaler.transform(X_clean)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit to data, then transform it in a single step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Revert scaled data back to the original space.

        This is primarily useful for debugging or for generating human‑readable
        feature values after a strategy has produced a signal.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X_arr = self._validate_input(X_scaled, allow_nan=False)
        return self.scaler.inverse_transform(X_arr)

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save(self, path: str) -> None:
        """Serialise the internal ``StandardScaler`` to ``path``."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a previously saved scaler.

        The returned instance is marked as fitted.
        """
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance

    # --------------------------------------------------------------------- #
    # Strategy‑related helpers
    # --------------------------------------------------------------------- #
    def generate_signal(
        self,
        X_scaled: pd.DataFrame | np.ndarray,
        entry_threshold: float = 0.5,
        exit_threshold: float = -0.5,
        *,
        confirm_window: int = 3,
        min_confirm: Optional[int] = None,
    ) -> np.ndarray:
        """Generate simple long/short signals from scaled features.

        The method interprets values above ``entry_threshold`` as a bullish
        entry condition and values below ``exit_threshold`` as a bearish exit
        condition.  To tighten entry quality, a confirmation filter can be
        applied: the signal must be present for a configurable number of
        consecutive periods before it is emitted.

        Parameters
        ----------
        X_scaled: pd.DataFrame | np.ndarray
            Scaled feature matrix.  Each column is treated independently.
        entry_threshold: float, default 0.5
            Minimum scaled value required for a long entry.
        exit_threshold: float, default -0.5
            Maximum scaled value required for a short exit.
        confirm_window: int, default 3
            Number of most recent periods to consider for confirmation.
        min_confirm: int | None
            Minimum number of periods within ``confirm_window`` that must
            satisfy the entry condition.  If ``None``, defaults to
            ``confirm_window`` (i.e., all periods must agree).

        Returns
        -------
        np.ndarray
            Array of signals with shape ``(n_samples,)`` where ``1`` denotes a
            long entry, ``-1`` denotes an exit/short, and ``0`` denotes no
            action.
        """
        X_arr = self._validate_input(X_scaled, allow_nan=False)
        if X_arr.ndim != 2:
            raise ValueError("X_scaled must be a 2‑dimensional array")
        # Basic signal based on thresholds
        raw_signal = np.where(X_arr > entry_threshold, 1, 0)
        raw_signal = np.where(X_arr < exit_threshold, -1, raw_signal)

        # Apply confirmation filter
        if confirm_window > 1:
            min_confirm = confirm_window if min_confirm is None else min_confirm
            # Rolling window using pandas for simplicity and edge‑case handling
            df = pd.DataFrame(raw_signal)
            # Count of positive entries in the window
            pos_counts = (
                (df == 1)
                .rolling(window=confirm_window, min_periods=confirm_window)
                .sum()
                .fillna(0)
                .astype(int)
                .values
            )
            # Count of negative entries in the window
            neg_counts = (
                (df == -1)
                .rolling(window=confirm_window, min_periods=confirm_window)
                .sum()
                .fillna(0)
                .astype(int)
                .values
            )
            confirmed = np.where(
                (pos_counts >= min_confirm) & (raw_signal == 1), 1, 0
            )
            confirmed = np.where(
                (neg_counts >= min_confirm) & (raw_signal == -1), -1, confirmed
            )
            raw_signal = confirmed

        return raw_signal.ravel()

    # --------------------------------------------------------------------- #
    # Internal utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def _validate_input(
        X: pd.DataFrame | np.ndarray, *, allow_nan: bool = True
    ) -> np.ndarray:
        """Convert input to ``np.ndarray`` and optionally enforce NaN checks.

        Parameters
        ----------
        X: pd.DataFrame | np.ndarray
            Input data.
        allow_nan: bool, default True
            If ``False``, a ``ValueError`` is raised when NaNs are present.

        Returns
        -------
        np.ndarray
            A contiguous ``float64`` array suitable for ``StandardScaler``.
        """
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(copy=False, dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)

        if not allow_nan and np.isnan(arr).any():
            raise ValueError("Input contains NaN values, which are not allowed")
        return arr