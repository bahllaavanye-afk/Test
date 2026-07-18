"""
LightGBM classifier — faster than XGBoost, often matches on financial data.
Includes SHAP explainability.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch

from app.ml.models.base_model import AbstractModel, EvalMetrics
from app.utils.logging import logger

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class LightGBMConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    max_depth: int = -1
    early_stopping_rounds: int = 50


class LightGBMClassifier(AbstractModel):
    """
    LightGBM binary classifier for direction prediction.
    Use LightGBMClassifier.from_config(LightGBMConfig()) to create.
    """
    model_type = "lightgbm"

    def __init__(self, config: LightGBMConfig | None = None):
        if config is not None and not isinstance(config, LightGBMConfig):
            raise ValueError("config must be a LightGBMConfig instance or None")
        self.config = config or LightGBMConfig()
        self._model: "lgb.Booster | None" = None
        self._feature_names: list[str] = []
        self._shap_explainer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input x must be a torch.Tensor")
        if self._model is None:
            raise RuntimeError("Model not trained yet")
        arr = x.numpy()
        if arr.ndim == 3:
            arr = arr[:, -1, :]  # use last timestep for flat features
        return torch.tensor(self._model.predict(arr), dtype=torch.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> dict:
        if not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be a numpy.ndarray")
        if not isinstance(y_train, np.ndarray):
            raise ValueError("y_train must be a numpy.ndarray")
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2‑dimensional array")
        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1‑dimensional array")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of rows in X_train must match length of y_train")
        if X_val is not None:
            if not isinstance(X_val, np.ndarray):
                raise ValueError("X_val must be a numpy.ndarray if provided")
            if X_val.ndim != 2:
                raise ValueError("X_val must be a 2‑dimensional array")
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError("X_val must have the same number of columns as X_train")
        if y_val is not None:
            if not isinstance(y_val, np.ndarray):
                raise ValueError("y_val must be a numpy.ndarray if provided")
            if y_val.ndim != 1:
                raise ValueError("y_val must be a 1‑dimensional array")
            if X_val is None:
                raise ValueError("y_val provided without X_val")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("Number of rows in X_val must match length of y_val")
        if feature_names is not None:
            if not isinstance(feature_names, list) or not all(isinstance(n, str) for n in feature_names):
                raise ValueError("feature_names must be a list of strings")
            if len(feature_names) != X_train.shape[1]:
                raise ValueError("Length of feature_names must match number of columns in X_train")

        if not HAS_LGB:
            logger.warning("lightgbm not installed. Install: pip install lightgbm")
            return {"error": "lightgbm not installed"}

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=self._feature_names)
        valid_sets = [train_set]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(val_set)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "min_child_samples": self.config.min_child_samples,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "max_depth": self.config.max_depth,
            "verbose": -1,
        }
        callbacks = [lgb.early_stopping(self.config.early_stopping_rounds), lgb.log_evaluation(50)]
        self._model = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        best_iter = self._model.best_iteration
        logger.info(f"LightGBM trained: best_iteration={best_iter}")
        return {"best_iteration": best_iter, "best_score": self._model.best_score}

    def train_epoch(self, loader: Iterable[Tuple[torch.Tensor, torch.Tensor]], optimizer, criterion) -> dict:
        # Collect all data and do a full LightGBM fit
        if not hasattr(loader, "__iter__"):
            raise ValueError("loader must be an iterable yielding (x, y) tuples")
        X, Y = [], []
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                raise ValueError("Each loader element must be a (x, y) pair")
            x, y = batch
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise ValueError("x and y must be torch.Tensor instances")
            arr = x.numpy()
            if arr.ndim == 3:
                arr = arr[:, -1, :]
            X.append(arr)
            Y.append(y.numpy())
        if not X:
            raise ValueError("loader yielded no data")
        X = np.vstack(X)
        Y = np.concatenate(Y)
        return self.fit(X, Y)

    def evaluate(self, loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> EvalMetrics:
        if self._model is None:
            return EvalMetrics(accuracy=0.5, auc=0.5, sharpe=0.0)
        if not hasattr(loader, "__iter__"):
            raise ValueError("loader must be an iterable yielding (x, y) tuples")
        X, Y = [], []
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                raise ValueError("Each loader element must be a (x, y) pair")
            x, y = batch
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise ValueError("x and y must be torch.Tensor instances")
            arr = x.numpy()
            if arr.ndim == 3:
                arr = arr[:, -1, :]
            X.append(arr)
            Y.append(y.numpy())
        if not X:
            raise ValueError("loader yielded no data")
        X = np.vstack(X)
        Y = np.concatenate(Y)
        preds = self._model.predict(X)
        acc = float(((preds > 0.5) == (Y > 0.5)).mean())
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(Y, preds))
        except Exception:
            auc = 0.5
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        imp = self._model.feature_importance(importance_type="gain")
        names = self._feature_names or self._model.feature_name()
        total = sum(imp) or 1
        return {n: round(float(v) / total, 4) for n, v in zip(names, imp)}

    def shap_values(self, X: np.ndarray) -> np.ndarray | None:
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy.ndarray")
        if X.ndim != 2:
            raise ValueError("X must be a 2‑dimensional array")
        if not HAS_SHAP or self._model is None:
            return None
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self._model)
        return self._shap_explainer.shap_values(X)

    def save(self, path: str, metadata: dict | None = None) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non‑empty string")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict if provided")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._model:
            self._model.save_model(path + ".lgb")
        meta = {"model_type": self.model_type, "feature_names": self._feature_names, **(metadata or {})}
        Path(path + ".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str) -> "LightGBMClassifier":
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non‑empty string")
        obj = cls()
        if not HAS_LGB:
            return obj
        obj._model = lgb.Booster(model_file=path + ".lgb")
        meta_path = Path(path + ".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            obj._feature_names = meta.get("feature_names", [])
        return obj