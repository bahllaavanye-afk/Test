"""
LightGBM classifier — faster than XGBoost, often matches on financial data.
Includes SHAP explainability.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional

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

    def __init__(self, config: Optional[LightGBMConfig] = None):
        self.config = config or LightGBMConfig()
        self._model: "lgb.Booster | None" = None
        self._feature_names: List[str] = []
        self._shap_explainer = None

    @staticmethod
    def _prepare_array(arr: np.ndarray) -> np.ndarray:
        """
        Convert a possibly 3‑dimensional array to 2‑dimensional by using the last timestep.
        Handles edge cases where the array is empty or does not have the expected dimensions.
        """
        if arr is None:
            raise ValueError("Input array is None")
        if arr.ndim == 0:
            raise ValueError("Input array has no dimensions")
        if arr.ndim == 3:
            if arr.shape[1] == 0:
                raise ValueError("3‑D input array has zero timesteps")
            arr = arr[:, -1, :]  # use last timestep for flat features
        return arr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not trained yet")
        if x is None:
            raise ValueError("Input tensor is None")
        arr = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        arr = self._prepare_array(arr)
        preds = self._model.predict(arr)
        return torch.tensor(preds, dtype=torch.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: List[str] | None = None,
    ) -> dict:
        if not HAS_LGB:
            logger.warning("lightgbm not installed. Install: pip install lightgbm")
            return {"error": "lightgbm not installed"}

        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data cannot be empty")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of training samples and labels must match")

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=self._feature_names)

        valid_sets = [train_set]
        if X_val is not None and y_val is not None:
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("Number of validation samples and labels must match")
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

    def train_epoch(self, loader, optimizer, criterion) -> dict:
        # Collect all data and do a full LightGBM fit
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        for x, y in loader:
            arr = x.numpy()
            arr = self._prepare_array(arr)
            X_list.append(arr)
            Y_list.append(y.numpy())
        if not X_list:
            raise ValueError("Data loader returned no batches")
        X = np.vstack(X_list)
        Y = np.concatenate(Y_list)
        return self.fit(X, Y)

    def evaluate(self, loader) -> EvalMetrics:
        if self._model is None:
            return EvalMetrics(accuracy=0.5, auc=0.5, sharpe=0.0)
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        for x, y in loader:
            arr = x.numpy()
            arr = self._prepare_array(arr)
            X_list.append(arr)
            Y_list.append(y.numpy())
        if not X_list:
            logger.warning("Evaluation loader is empty; returning default metrics")
            return EvalMetrics(accuracy=0.5, auc=0.5, sharpe=0.0)
        X = np.vstack(X_list)
        Y = np.concatenate(Y_list)
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
        # Guard against mismatched lengths
        if len(imp) != len(names):
            logger.warning(
                "Feature importance length (%d) does not match number of feature names (%d); "
                "truncating to the smaller size",
                len(imp),
                len(names),
            )
            min_len = min(len(imp), len(names))
            imp = imp[:min_len]
            names = names[:min_len]
        total = float(sum(imp)) or 1.0
        return {n: round(float(v) / total, 4) for n, v in zip(names, imp)}

    def shap_values(self, X: np.ndarray) -> np.ndarray | None:
        if not HAS_SHAP or self._model is None:
            return None
        if X is None or X.size == 0:
            logger.warning("Empty input provided to shap_values; returning None")
            return None
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self._model)
        return self._shap_explainer.shap_values(X)

    def save(self, path: str, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._model:
            self._model.save_model(path + ".lgb")
        meta = {"model_type": self.model_type, "feature_names": self._feature_names, **(metadata or {})}
        Path(path + ".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str) -> "LightGBMClassifier":
        obj = cls()
        if not HAS_LGB:
            return obj
        obj._model = lgb.Booster(model_file=path + ".lgb")
        meta_path = Path(path + ".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            obj._feature_names = meta.get("feature_names", [])
        return obj