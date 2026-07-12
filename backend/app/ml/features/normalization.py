import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import unittest
import tempfile
import os


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


class TestFeatureScalerEdgeCases(unittest.TestCase):
    def test_transform_without_fit_raises(self):
        scaler = FeatureScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(np.array([[1, 2], [3, 4]]))

    def test_fit_transform_with_empty_array(self):
        scaler = FeatureScaler()
        empty_array = np.empty((0, 2))
        # Fit on empty should not raise but will mark as fitted; StandardScaler expects at least 1 sample,
        # so we expect a ValueError from sklearn.
        with self.assertRaises(ValueError):
            scaler.fit_transform(empty_array)

    def test_save_and_load_preserves_state(self):
        scaler = FeatureScaler()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaler.fit(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "scaler.pkl")
            scaler.save(file_path)

            # Load a new instance
            loaded_scaler = FeatureScaler.load(file_path)

            # Verify that the loaded scaler produces the same transformation
            transformed_original = scaler.transform(data)
            transformed_loaded = loaded_scaler.transform(data)
            np.testing.assert_allclose(transformed_original, transformed_loaded)

    def test_fit_transform_single_sample(self):
        scaler = FeatureScaler()
        single_sample = np.array([[5.0, -3.0]])
        transformed = scaler.fit_transform(single_sample)
        # With a single sample, StandardScaler will produce zeros after scaling
        expected = np.zeros_like(single_sample)
        np.testing.assert_allclose(transformed, expected)


if __name__ == "__main__":
    unittest.main()