import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around ``StandardScaler`` with convenient save/load methods for inference.

    The scaler can be fitted on a pandas ``DataFrame`` or a NumPy ``ndarray``.  After fitting,
    the internal ``StandardScaler`` instance is marked as ready for transformation.  The
    object can be persisted to disk using :meth:`save` and later restored with :meth:`load`.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        """Fit the scaler to ``X``."""
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform ``X`` using the fitted scaler.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted scaler to ``path``.

        The target directory is created automatically if it does not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureScaler":
        """Load a previously saved scaler from ``path``.

        The returned instance is marked as fitted.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No scaler file found at {path}")
        instance = cls()
        with path.open("rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "unfitted"
        return f"{self.__class__.__name__}({status})"