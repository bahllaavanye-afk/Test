import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field, validator


class FeatureScaler:
    """Wrapper around StandardScaler with save/load for inference."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        """Fit the scaler to the data."""
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted — call fit() first")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        """Persist the fitted scaler to a file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a persisted scaler from a file."""
        instance = cls()
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance.fitted = True
        return instance


class FeatureScalerSaveConfig(BaseModel):
    """Configuration for saving a FeatureScaler."""

    path: str = Field(
        ...,
        description="Filesystem path where the scaler will be saved.",
        example="/models/feature_scaler.pkl",
    )

    @validator("path")
    def ensure_parent_dir_exists(cls, v: str) -> str:
        """Validate that the parent directory exists or can be created."""
        parent = Path(v).parent
        if not parent.exists():
            # Attempt to create the directory to provide early feedback.
            parent.mkdir(parents=True, exist_ok=True)
        return v


class FeatureScalerLoadConfig(BaseModel):
    """Configuration for loading a FeatureScaler."""

    path: str = Field(
        ...,
        description="Filesystem path from which the scaler will be loaded.",
        example="/models/feature_scaler.pkl",
    )

    @validator("path")
    def ensure_file_exists(cls, v: str) -> str:
        """Validate that the specified file exists."""
        if not Path(v).is_file():
            raise ValueError(f"Scaler file not found at path: {v}")
        return v

__all__ = [
    "FeatureScaler",
    "FeatureScalerSaveConfig",
    "FeatureScalerLoadConfig",
]