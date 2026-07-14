import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around ``StandardScaler`` with persistence utilities and simple
    signal helpers for trading strategies.

    The class is deliberately lightweight: it only stores the underlying
    ``StandardScaler`` instance and a ``fitted`` flag.  Additional helpers are
    provided to tighten entry conditions, add confirmation filters, and improve
    exit logic based on the normalized features.

    Attributes
    ----------
    scaler: StandardScaler
        The scikit‑learn scaler used for mean‑variance normalization.
    fitted: bool
        Indicates whether ``fit`` has been called.
    clip_threshold: float
        Absolute z‑score limit applied after transformation. Values beyond this
        threshold are clipped to reduce the impact of outliers on downstream
        strategy decisions.
    """

    def __init__(self, clip_threshold: float = 3.0):
        self.scaler = StandardScaler()
        self.fitted = False
        self.clip_threshold = clip_threshold

    # --------------------------------------------------------------------- #
    # Core scaler API
    # --------------------------------------------------------------------- #
    def fit(self, X: Union[pd.DataFrame, np.ndarray, Any]) -> "FeatureScaler":
        """Fit the internal ``StandardScaler`` to ``X``."""
        self.scaler.fit(self._to_numpy(X))
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray, Any]) -> np.ndarray:
        """Transform ``X`` using the fitted scaler and clip extreme values.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        transformed = self.scaler.transform(self._to_numpy(X))
        # Clip extreme z‑scores to limit outlier influence on signals.
        if self.clip_threshold is not None:
            np.clip(transformed, -self.clip_threshold, self.clip_threshold, out=transformed)
        return transformed

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray, Any]) -> np.ndarray:
        """Convenience method that fits the scaler and returns the transformed data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray, Any]) -> np.ndarray:
        """Revert a previously transformed array back to the original space."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.inverse_transform(self._to_numpy(X))

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #
    def save(self, path: Union[str, Path]) -> None:
        """Serialise the underlying ``StandardScaler`` to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureScaler":
        """Load a previously saved scaler from ``path``."""
        path = Path(path)
        instance = cls()
        with path.open("rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance

    # --------------------------------------------------------------------- #
    # Strategy‑specific helpers
    # --------------------------------------------------------------------- #
    def entry_signal(
        self,
        X: Union[pd.DataFrame, np.ndarray, Any],
        threshold: float = 1.5,
        window: int = 5,
    ) -> np.ndarray:
        """Generate a boolean entry mask.

        The mask is ``True`` when the normalized feature exceeds ``threshold`` and
        its rolling mean over ``window`` periods is also positive, providing a
        simple confirmation filter.

        Parameters
        ----------
        X : DataFrame or ndarray
            Raw feature matrix to be evaluated.
        threshold : float, default 1.5
            Minimum normalized value required for a potential entry.
        window : int, default 5
            Length of the rolling window used for confirmation.

        Returns
        -------
        np.ndarray
            Boolean array where ``True`` indicates a qualified entry signal.
        """
        norm = self.transform(X)
        if norm.ndim == 1:
            norm = norm[:, None]

        # Rolling mean using pandas for convenience while preserving shape
        df = pd.DataFrame(norm)
        rolling_mean = df.rolling(window, min_periods=1).mean().values

        return (norm > threshold) & (rolling_mean > 0)

    def exit_signal(
        self,
        X: Union[pd.DataFrame, np.ndarray, Any],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Generate a boolean exit mask.

        An exit is signalled when the normalized feature falls below ``threshold``
        or crosses zero, encouraging a timely profit‑taking or risk‑mitigation
        action.

        Parameters
        ----------
        X : DataFrame or ndarray
            Raw feature matrix to be evaluated.
        threshold : float, default 0.5
            Upper bound for the normalized value to trigger an exit.

        Returns
        -------
        np.ndarray
            Boolean array where ``True`` indicates a qualified exit signal.
        """
        norm = self.transform(X)
        if norm.ndim == 1:
            norm = norm[:, None]

        # Exit when value is low or sign changes (crosses zero)
        cross_zero = np.sign(norm[:, :-1]) != np.sign(norm[:, 1:])
        # Pad to match original length
        cross_zero = np.concatenate([cross_zero, np.zeros((norm.shape[0], 1), dtype=bool)], axis=1)

        return (norm < threshold) | cross_zero

    # --------------------------------------------------------------------- #
    # Internal utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def _to_numpy(X: Union[pd.DataFrame, np.ndarray, Any]) -> np.ndarray:
        """Convert input to a 2‑D ``np.ndarray`` suitable for scikit‑learn."""
        if isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, np.ndarray):
            return X.reshape(-1, X.shape[-1]) if X.ndim == 1 else X
        raise TypeError(
            f"Unsupported input type {type(X)}; expected pandas DataFrame or numpy ndarray."
        )