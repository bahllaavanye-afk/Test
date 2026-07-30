import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around ``StandardScaler`` with persistence and simple signal utilities.

    The class is deliberately lightweight: it only adds convenience methods on top of
    ``StandardScaler`` while keeping the original API unchanged.  The additional
    ``generate_signal`` method is intended for the *mean_rev_20_2* strategy and
    provides tighter entry conditions, confirmation filtering, and a clearer exit
    rule based on the scaled feature values.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    # --------------------------------------------------------------------- #
    # Core scaler interface
    # --------------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        """Fit the internal ``StandardScaler`` and mark the instance as ready."""
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler.

        Raises:
            RuntimeError: If ``fit`` has not been called first.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit the scaler on ``X`` and return the transformed values."""
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
    # Strategy‑specific utilities
    # --------------------------------------------------------------------- #
    def _confirm(
        self,
        series: pd.Series,
        window: int,
        condition: Callable[[float], bool],
    ) -> pd.Series:
        """Apply a rolling confirmation filter.

        For each element, the function checks that the ``condition`` holds for the
        current value *and* the previous ``window``‑1 values.  The result is a
        boolean series where ``True`` indicates a confirmed signal.
        """
        if window <= 1:
            return series.apply(condition)

        # Rolling apply is slower but keeps the logic explicit and pandas‑centric.
        def _window_check(idx: int) -> bool:
            start = max(0, idx - window + 1)
            return all(condition(v) for v in series.iloc[start : idx + 1])

        return pd.Series([_window_check(i) for i in range(len(series))], index=series.index)

    def generate_signal(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        entry_threshold: float = 1.0,
        exit_threshold: float = 0.5,
        confirmation_window: int = 3,
        column: str | int = 0,
    ) -> pd.Series:
        """Create a simple long‑only signal based on the scaled feature.

        The algorithm works as follows:

        1. **Normalization** – the data is scaled using the fitted ``StandardScaler``.
        2. **Entry condition** – a scaled value greater than ``entry_threshold`` triggers a
           potential entry.
        3. **Confirmation filter** – the entry must be sustained for
           ``confirmation_window`` consecutive periods.
        4. **Exit condition** – once a position is open, it is exited when the scaled
           value falls below ``exit_threshold``.

        Parameters
        ----------
        X :
            Raw feature matrix.  If a ``DataFrame`` is supplied, ``column`` selects the
            feature to be used for the signal; otherwise the first column of the
            ``ndarray`` is used.
        entry_threshold :
            Z‑score above which an entry is considered.
        exit_threshold :
            Z‑score below which an open position is closed.
        confirmation_window :
            Number of consecutive periods the entry condition must hold.
        column :
            Column identifier (name or integer) for ``DataFrame`` inputs.

        Returns
        -------
        pd.Series
            A series indexed like the input containing ``1`` for entry,
            ``-1`` for exit, and ``0`` otherwise.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() before generating signals")

        # ----------------------------------------------------------------- #
        # 1. Normalisation
        # ----------------------------------------------------------------- #
        if isinstance(X, pd.DataFrame):
            data = X[[column]] if isinstance(column, str) else X.iloc[:, column]
        else:
            data = pd.DataFrame(X[:, column] if isinstance(column, int) else X)

        scaled = pd.Series(self.transform(data).ravel(), index=data.index)

        # ----------------------------------------------------------------- #
        # 2. Entry & confirmation
        # ----------------------------------------------------------------- #
        entry_cond = lambda v: v > entry_threshold
        confirmed_entry = self._confirm(scaled, confirmation_window, entry_cond)

        # ----------------------------------------------------------------- #
        # 3. Exit logic
        # ----------------------------------------------------------------- #
        exit_cond = lambda v: v < exit_threshold

        # Build the raw signal: 1 for entry, -1 for exit, 0 otherwise.
        raw_signal = pd.Series(0, index=scaled.index, dtype=int)
        raw_signal[confirmed_entry] = 1
        raw_signal[scaled.apply(exit_cond)] = -1

        # ----------------------------------------------------------------- #
        # 4. Enforce position persistence (no overlapping entries)
        # ----------------------------------------------------------------- #
        signal = pd.Series(0, index=scaled.index, dtype=int)
        position_open = False
        for idx in scaled.index:
            if not position_open and raw_signal.loc[idx] == 1:
                signal.loc[idx] = 1
                position_open = True
            elif position_open and raw_signal.loc[idx] == -1:
                signal.loc[idx] = -1
                position_open = False
            # else keep signal as 0 (no change)

        return signal