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
# Unit tests for edge cases
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import unittest
    import tempfile
    import os

    class TestFeatureScalerEdgeCases(unittest.TestCase):
        def test_transform_without_fit_raises(self):
            scaler = FeatureScaler()
            data = np.array([[1.0, 2.0], [3.0, 4.0]])
            with self.assertRaises(RuntimeError) as cm:
                scaler.transform(data)
            self.assertIn("Scaler not fitted", str(cm.exception))

        def test_fit_with_empty_input_raises(self):
            scaler = FeatureScaler()
            empty_array = np.empty((0, 2))
            with self.assertRaises(ValueError):
                scaler.fit(empty_array)

            empty_df = pd.DataFrame(columns=["a", "b"])
            with self.assertRaises(ValueError):
                scaler.fit(empty_df)

        def test_save_and_load_preserves_transformation(self):
            scaler = FeatureScaler()
            data = pd.DataFrame({"x": [10, 20, 30], "y": [1, 2, 3]})
            scaler.fit(data)

            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "nested", "scaler.pkl")
                scaler.save(file_path)

                # Ensure the file was created
                self.assertTrue(Path(file_path).is_file())

                # Load and compare transformations
                loaded_scaler = FeatureScaler.load(file_path)
                transformed_original = scaler.transform(data)
                transformed_loaded = loaded_scaler.transform(data)
                np.testing.assert_allclose(transformed_original, transformed_loaded)

        def test_non_numeric_input_raises(self):
            scaler = FeatureScaler()
            df = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "c"]})
            with self.assertRaises(ValueError):
                scaler.fit(df)

    unittest.main(argv=["first-arg-is-ignored"], exit=False)