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
# Unit tests for edge cases and boundary conditions
# ----------------------------------------------------------------------
import unittest
import tempfile
import os


class TestFeatureScalerEdgeCases(unittest.TestCase):
    def test_transform_without_fit_raises(self):
        """Calling transform before fit should raise a RuntimeError."""
        scaler = FeatureScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(RuntimeError) as cm:
            scaler.transform(X)
        self.assertIn("Scaler not fitted", str(cm.exception))

    def test_fit_transform_empty_input_raises(self):
        """Fitting on an empty DataFrame should raise a ValueError from sklearn."""
        scaler = FeatureScaler()
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            scaler.fit_transform(empty_df)

    def test_save_load_consistency(self):
        """Saving and loading should preserve the scaler state and allow transformation."""
        scaler = FeatureScaler()
        X = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        scaler.fit(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "scaler.pkl")
            scaler.save(file_path)

            # Ensure the file was created
            self.assertTrue(Path(file_path).exists())

            # Load and verify transformation consistency
            loaded_scaler = FeatureScaler.load(file_path)
            X_new = np.array([[5.0, 15.0], [35.0, 45.0]])
            transformed_original = scaler.transform(X_new)
            transformed_loaded = loaded_scaler.transform(X_new)

            # Results should be identical (within floating point tolerance)
            np.testing.assert_allclose(transformed_original, transformed_loaded)


if __name__ == "__main__":
    unittest.main()