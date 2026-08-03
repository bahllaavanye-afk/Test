import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import unittest
import tempfile
import shutil


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
        scaler = FeatureScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(np.array([[1, 2], [3, 4]]))

    def test_fit_transform_constant_feature(self):
        # Constant column should result in zeros after scaling
        data = pd.DataFrame({"const": [5, 5, 5], "var": [1, 2, 3]})
        scaler = FeatureScaler()
        transformed = scaler.fit_transform(data)
        # First column (constant) should be all zeros
        self.assertTrue(np.allclose(transformed[:, 0], 0))
        # Second column should be standardized correctly
        expected = StandardScaler().fit_transform(data)
        self.assertTrue(np.allclose(transformed, expected))

    def test_save_load_preserves_state(self):
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        scaler = FeatureScaler().fit(data)

        tmp_dir = tempfile.mkdtemp()
        try:
            file_path = Path(tmp_dir) / "scaler.pkl"
            scaler.save(str(file_path))

            loaded_scaler = FeatureScaler.load(str(file_path))
            # Ensure loaded scaler produces same transformation
            original_transformed = scaler.transform(data)
            loaded_transformed = loaded_scaler.transform(data)
            self.assertTrue(np.allclose(original_transformed, loaded_transformed))
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()