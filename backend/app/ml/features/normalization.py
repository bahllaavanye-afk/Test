import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance


# ==========================
# Unit tests for FeatureScaler
# ==========================
import tempfile
import os
import pytest


def test_transform_before_fit_raises():
    """Calling transform before fit should raise RuntimeError."""
    scaler = FeatureScaler()
    with pytest.raises(RuntimeError, match="Scaler not fitted"):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_fit_transform_empty_input_raises():
    """Fitting on an empty array should raise a ValueError from sklearn."""
    scaler = FeatureScaler()
    empty_array = np.empty((0, 2))
    with pytest.raises(ValueError):
        scaler.fit_transform(empty_array)


def test_save_and_load_preserves_scaler_behavior():
    """After saving and loading, the scaler should produce identical transformations."""
    scaler = FeatureScaler()
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    transformed_original = scaler.fit_transform(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "scaler.pkl")
        scaler.save(file_path)

        # Load a new instance from disk
        loaded_scaler = FeatureScaler.load(file_path)
        transformed_loaded = loaded_scaler.transform(data)

        # The transformed outputs should be numerically identical
        np.testing.assert_allclose(transformed_original, transformed_loaded, rtol=1e-7, atol=0)