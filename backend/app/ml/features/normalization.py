import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference and basic signal utilities."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        """Fit the internal StandardScaler."""
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit the scaler then transform the data."""
        return self.fit(X).transform(X)

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

    # -------------------------------------------------------------------------
    # Strategy‑related utilities
    # -------------------------------------------------------------------------

    def _to_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Ensure input is a DataFrame for rolling operations."""
        if isinstance(X, pd.DataFrame):
            return X
        # Convert numpy array to DataFrame; columns are numbered
        return pd.DataFrame(X)

    def generate_entry_signal(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        threshold: float = 0.5,
        confirmation_window: int = 3,
        confirmation_delta: float = 0.1,
    ) -> pd.Series:
        """
        Produce a binary entry signal based on normalized feature values.

        Conditions:
        1. Current value must cross above ``threshold`` (i.e., be greater than it).
        2. A short‑term rolling mean must also be above ``threshold + confirmation_delta``
           to filter out fleeting spikes.

        Returns a pandas Series of 0/1 where 1 indicates a valid entry point.
        """
        df = self._to_dataframe(X)
        # Primary crossing condition
        primary = (df > threshold).astype(int)

        # Confirmation using rolling mean (centered on the current row)
        rolling_mean = df.rolling(window=confirmation_window, min_periods=1).mean()
        confirmation = (rolling_mean > (threshold + confirmation_delta)).astype(int)

        # Combine both conditions
        entry_signal = (primary & confirmation).iloc[:, 0] if primary.shape[1] == 1 else primary.mul(confirmation).sum(axis=1).gt(0).astype(int)
        return entry_signal

    def generate_exit_signal(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        threshold: float = 0.2,
        confirmation_window: int = 3,
        confirmation_delta: float = 0.1,
    ) -> pd.Series:
        """
        Produce a binary exit signal.

        Conditions:
        1. Current value falls below ``threshold``.
        2. The rolling mean over ``confirmation_window`` rows is also below
           ``threshold - confirmation_delta`` to avoid premature exits.

        Returns a pandas Series of 0/1 where 1 indicates a valid exit point.
        """
        df = self._to_dataframe(X)
        primary = (df < threshold).astype(int)
        rolling_mean = df.rolling(window=confirmation_window, min_periods=1).mean()
        confirmation = (rolling_mean < (threshold - confirmation_delta)).astype(int)

        exit_signal = (primary & confirmation).iloc[:, 0] if primary.shape[1] == 1 else primary.mul(confirmation).sum(axis=1).gt(0).astype(int)
        return exit_signal

    def evaluate_position(
        self,
        entry_signal: pd.Series,
        exit_signal: pd.Series,
        *,
        max_holding_period: int | None = None,
    ) -> pd.Series:
        """
        Derive a position series (1 = in‑position, 0 = flat) from entry/exit signals.

        The logic respects:
        * Entry signal opens a position.
        * Exit signal closes the position.
        * Optional ``max_holding_period`` forces an exit after the given number of
          periods even if no explicit exit signal is received.

        Parameters
        ----------
        entry_signal: pd.Series
            Binary series indicating entry points.
        exit_signal: pd.Series
            Binary series indicating exit points.
        max_holding_period: int | None
            Maximum number of periods to stay in a trade. ``None`` disables the limit.

        Returns
        -------
        pd.Series
            Binary series where 1 denotes an active position.
        """
        # Ensure signals are aligned
        entry = entry_signal.reindex_like(exit_signal).fillna(0).astype(int)
        exit_ = exit_signal.reindex_like(entry_signal).fillna(0).astype(int)

        position = pd.Series(0, index=entry.index, dtype=int)
        holding_counter = 0

        for idx in entry.index:
            if position[idx - 1] if idx != entry.index[0] else 0:
                # We are already in a position
                holding_counter += 1
                if exit_[idx] or (max_holding_period is not None and holding_counter >= max_holding_period):
                    position[idx] = 0
                    holding_counter = 0
                else:
                    position[idx] = 1
            else:
                # Not in a position
                if entry[idx]:
                    position[idx] = 1
                    holding_counter = 1
                else:
                    position[idx] = 0

        return position

    def reset(self) -> None:
        """Reset the scaler to an unfitted state without discarding the object."""
        self.scaler = StandardScaler()
        self.fitted = False