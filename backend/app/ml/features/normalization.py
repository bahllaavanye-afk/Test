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


class TestFeatureScaler(unittest.TestCase):
    def test_transform_without_fit_raises(self):
        """Calling transform before fit should raise RuntimeError."""
        scaler = FeatureScaler()
        X = np.array([[1, 2], [3, 4]])
        with self.assertRaises(RuntimeError):
            scaler.transform(X)

    def test_fit_with_empty_input_raises(self):
        """Fitting on an empty array should propagate sklearn's ValueError."""
        scaler = FeatureScaler()
        X_empty = np.empty((0, 2))
        with self.assertRaises(ValueError):
            scaler.fit(X_empty)

    def test_save_and_load_preserves_state(self):
        """After saving and loading, the scaler should retain its parameters and be usable."""
        scaler = FeatureScaler()
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        scaler.fit(X)

        # Use a temporary file for persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "scaler.pkl")
            scaler.save(file_path)

            # Load a new instance
            loaded_scaler = FeatureScaler.load(file_path)

            # Verify that the loaded instance is marked as fitted
            self.assertTrue(loaded_scaler.fitted)

            # Transform using both original and loaded scaler should give identical results
            original_transformed = scaler.transform(X)
            loaded_transformed = loaded_scaler.transform(X)
            np.testing.assert_allclose(original_transformed, loaded_transformed)

    def test_single_sample_boundary(self):
        """Scaling a single-sample input (shape 1, n) should not error and return a zero vector."""
        scaler = FeatureScaler()
        X = np.array([[10.0, -5.0, 3.0]])
        scaler.fit(X)
        transformed = scaler.transform(X)
        # For a single sample, StandardScaler centers to zero and scales to unit variance (resulting in zeros)
        np.testing.assert_allclose(transformed, np.zeros_like(transformed))


if __name__ == "__main__":
    unittest.main()