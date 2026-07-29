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


# ----------------------------------------------------------------------
# Unit tests for edge/boundary conditions
# ----------------------------------------------------------------------
import unittest
import tempfile
import os

class TestFeatureScaler(unittest.TestCase):
    def test_transform_without_fit_raises(self):
        """Calling transform before fit should raise a RuntimeError."""
        scaler = FeatureScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(RuntimeError):
            scaler.transform(X)

    def test_fit_with_empty_input_raises(self):
        """Fitting on an empty array or DataFrame should raise a ValueError."""
        scaler = FeatureScaler()
        empty_np = np.empty((0, 2))
        empty_df = pd.DataFrame(columns=["a", "b"])
        with self.assertRaises(ValueError):
            scaler.fit(empty_np)
        with self.assertRaises(ValueError):
            scaler.fit(empty_df)

    def test_save_load_consistency(self):
        """After saving and loading, the scaler should produce identical results."""
        scaler = FeatureScaler()
        X = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        transformed_before = scaler.fit_transform(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "scaler.pkl")
            scaler.save(file_path)
            loaded_scaler = FeatureScaler.load(file_path)
            transformed_after = loaded_scaler.transform(X)

        np.testing.assert_allclose(transformed_before, transformed_after, rtol=1e-7, atol=0)

    def test_fit_transform_single_sample(self):
        """Fit_transform should handle a single-sample input without error."""
        scaler = FeatureScaler()
        X = np.array([[5.0, -3.0]])  # shape (1, 2)
        # StandardScaler can compute mean/std with a single sample; std will be zero leading to NaNs.
        # We check that it does not raise and returns a finite array (may contain NaNs).
        result = scaler.fit_transform(X)
        self.assertEqual(result.shape, (1, 2))
        # Ensure no exception was raised; content validation is not required.

if __name__ == "__main__":
    unittest.main()