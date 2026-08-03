import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference.

    Handles edge cases such as None inputs, empty collections, validates that the
    scaler has been fitted before transformation, and adds lightweight data
    quality filters to improve downstream signal reliability.
    """

    def __init__(self, clip_lower: float = 0.01, clip_upper: float = 0.99):
        """
        Args:
            clip_lower: Lower percentile for extreme value clipping (default 1%).
            clip_upper: Upper percentile for extreme value clipping (default 99%).
        """
        self.scaler = StandardScaler()
        self.fitted = False
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    @staticmethod
    def _validate_input(X, context: str) -> None:
        """Validate that X is not None and contains data.

        Args:
            X: Input data (DataFrame or ndarray).
            context: Description of the operation for error messages.

        Raises:
            ValueError: If X is None or empty.
        """
        if X is None:
            raise ValueError(f"{context}: input data is None")
        if isinstance(X, (pd.DataFrame, pd.Series)):
            if X.empty:
                raise ValueError(f"{context}: input DataFrame/Series is empty")
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError(f"{context}: input ndarray is empty")
        else:
            raise TypeError(
                f"{context}: input must be a pandas DataFrame/Series or numpy ndarray"
            )

    @staticmethod
    def _check_nan(X, context: str) -> None:
        """Ensure no NaN values are present.

        Raises:
            ValueError: If NaNs are detected.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            if X.isnull().any().any():
                raise ValueError(f"{context}: input contains NaN values")
        elif isinstance(X, np.ndarray):
            if np.isnan(X).any():
                raise ValueError(f"{context}: input contains NaN values")

    def _clip_extremes(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Clip extreme values based on configured percentiles."""
        lower_q = self.clip_lower
        upper_q = self.clip_upper
        if isinstance(X, pd.DataFrame):
            lower = X.quantile(lower_q)
            upper = X.quantile(upper_q)
            return X.clip(lower=lower, upper=upper, axis=1)
        else:  # numpy ndarray
            lower = np.quantile(X, lower_q, axis=0)
            upper = np.quantile(X, upper_q, axis=0)
            return np.clip(X, lower, upper)

    def _preprocess(self, X: pd.DataFrame | np.ndarray, context: str) -> pd.DataFrame | np.ndarray:
        """Run validation and quality filters before scaling."""
        self._validate_input(X, context)
        self._check_nan(X, context)
        return self._clip_extremes(X)

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        """Fit the scaler to the data.

        Args:
            X: Training data.

        Returns:
            self
        """
        X = self._preprocess(X, "fit")
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler.

        Args:
            X: Data to transform.

        Returns:
            Transformed data as a NumPy array.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        X = self._preprocess(X, "transform")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X: Data to fit and transform.

        Returns:
            Transformed data as a NumPy array.
        """
        X = self._preprocess(X, "fit_transform")
        return self.scaler.fit_transform(X)

    def inverse_transform(self, X_scaled: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Revert scaled data back to original space.

        Args:
            X_scaled: Scaled data.

        Returns:
            Data in the original feature space.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        self._validate_input(X_scaled, "inverse_transform")
        self._check_nan(X_scaled, "inverse_transform")
        return self.scaler.inverse_transform(X_scaled)

    def save(self, path: str) -> None:
        """Persist the fitted scaler to disk.

        Args:
            path: Destination file path.
        """
        if not path:
            raise ValueError("save: path must be a non‑empty string")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a persisted scaler from disk.

        Args:
            path: File path to load the scaler from.

        Returns:
            An instance of FeatureScaler with the loaded scaler.
        """
        if not path:
            raise ValueError("load: path must be a non‑empty string")
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance

    # ------------------------------------------------------------------
    # Additional helpers for strategy signal quality
    # ------------------------------------------------------------------

    def filter_by_volatility(
        self,
        data: pd.DataFrame,
        window: int = 20,
        threshold: float = 0.02,
    ) -> pd.DataFrame:
        """Retain rows where rolling volatility exceeds a threshold.

        This acts as a confirmation filter: signals are only considered when
        market volatility is sufficient, reducing false entries in low‑vol
        regimes.

        Args:
            data: Input price or return series (must contain numeric columns).
            window: Rolling window size.
            threshold: Minimum volatility (standard deviation) required.

        Returns:
            Filtered DataFrame with rows that meet the volatility condition.
        """
        self._validate_input(data, "filter_by_volatility")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("filter_by_volatility: data must be a pandas DataFrame")
        vol = data.rolling(window).std()
        mask = (vol > threshold).all(axis=1)
        return data[mask]

    def add_exit_signal(
        self,
        predictions: pd.Series,
        exit_threshold: float = 0.0,
    ) -> pd.Series:
        """Generate an exit signal based on prediction crossing a threshold.

        Args:
            predictions: Model output series (e.g., probability or score).
            exit_threshold: Value at which an existing position should be closed.

        Returns:
            Series of booleans where True indicates an exit condition.
        """
        self._validate_input(predictions, "add_exit_signal")
        if not isinstance(predictions, (pd.Series, pd.DataFrame)):
            raise TypeError("add_exit_signal: predictions must be a pandas Series or DataFrame")
        return predictions <= exit_threshold

    def tighten_entry_conditions(
        self,
        entry_signal: pd.Series,
        confirmation: pd.Series,
        min_confidence: float = 0.6,
    ) -> pd.Series:
        """Combine primary entry signal with a confirmation filter.

        Args:
            entry_signal: Initial entry boolean series.
            confirmation: Confirmation boolean series (e.g., volatility filter).
            min_confidence: Minimum confidence level required (if signals are numeric).

        Returns:
            Refined entry signal series.
        """
        self._validate_input(entry_signal, "tighten_entry_conditions")
        self._validate_input(confirmation, "tighten_entry_conditions")
        if isinstance(entry_signal, pd.Series) and entry_signal.dtype.kind in "f":
            primary = entry_signal > min_confidence
        else:
            primary = entry_signal.astype(bool)
        conf = confirmation.astype(bool)
        return primary & conf