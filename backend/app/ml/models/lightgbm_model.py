"""
LightGBM classifier — faster than XGBoost, often matches on financial data.
Includes SHAP explainability.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
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
        self.config = config or LightGBMConfig()
        self._model: "lgb.Booster | None" = None
        self._feature_names: list[str] = []
        self._shap_explainer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not trained yet")
        try:
            arr = x.numpy() if isinstance(x, torch.Tensor) else x
            if arr.ndim == 3:
                arr = arr[:, -1, :]  # use last timestep for flat features
            preds = self._model.predict(arr)
            return torch.tensor(preds, dtype=torch.float32)
        except Exception as e:
            logger.exception(f"Forward pass failed: {e}")
            raise

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> dict:
        if not HAS_LGB:
            logger.warning("lightgbm not installed. Install: pip install lightgbm")
            return {"error": "lightgbm not installed"}

        try:
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
        except (ValueError, lgb.basic.LightGBMError) as e:
            logger.exception(f"LightGBM training failed due to invalid input: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.exception(f"Unexpected error during LightGBM training: {e}")
            return {"error": str(e)}

    def train_epoch(self, loader, optimizer, criterion) -> dict:
        try:
            X, Y = [], []
            for x, y in loader:
                arr = x.numpy()
                if arr.ndim == 3:
                    arr = arr[:, -1, :]
                X.append(arr)
                Y.append(y.numpy())
            X = np.vstack(X)
            Y = np.concatenate(Y)
            return self.fit(X, Y)
        except Exception as e:
            logger.exception(f"train_epoch failed: {e}")
            return {"error": str(e)}

    def evaluate(self, loader) -> EvalMetrics:
        if self._model is None:
            logger.warning("Evaluation called before model is trained")
            return EvalMetrics(accuracy=0.5, auc=0.5, sharpe=0.0)
        try:
            X, Y = [], []
            for x, y in loader:
                arr = x.numpy()
                if arr.ndim == 3:
                    arr = arr[:, -1, :]
                X.append(arr)
                Y.append(y.numpy())
            X = np.vstack(X)
            Y = np.concatenate(Y)
            preds = self._model.predict(X)
            acc = float(((preds > 0.5) == (Y > 0.5)).mean())
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(Y, preds))
            except Exception as e:
                logger.exception(f"AUC calculation failed: {e}")
                auc = 0.5
            return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)
        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            return EvalMetrics(accuracy=0.5, auc=0.5, sharpe=0.0)

    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            logger.warning("Feature importance requested before model is trained")
            return {}
        try:
            imp = self._model.feature_importance(importance_type="gain")
            names = self._feature_names or self._model.feature_name()
            total = sum(imp) or 1
            return {n: round(float(v) / total, 4) for n, v in zip(names, imp)}
        except Exception as e:
            logger.exception(f"Feature importance extraction failed: {e}")
            return {}

    def shap_values(self, X: np.ndarray) -> np.ndarray | None:
        if not HAS_SHAP or self._model is None:
            logger.warning("SHAP values requested but SHAP library or model is unavailable")
            return None
        try:
            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(self._model)
            return self._shap_explainer.shap_values(X)
        except Exception as e:
            logger.exception(f"SHAP value computation failed: {e}")
            return None

    def save(self, path: str, metadata: dict | None = None) -> None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if self._model:
                self._model.save_model(path + ".lgb")
            meta = {"model_type": self.model_type, "feature_names": self._feature_names, **(metadata or {})}
            Path(path + ".json").write_text(json.dumps(meta, indent=2))
            logger.info(f"Model saved to {path}")
        except OSError as e:
            logger.exception(f"Failed to save model files: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during model save: {e}")
            raise

    @classmethod
    def load(cls, path: str) -> "LightGBMClassifier":
        obj = cls()
        if not HAS_LGB:
            logger.warning("lightgbm not installed; cannot load model")
            return obj
        try:
            obj._model = lgb.Booster(model_file=path + ".lgb")
        except FileNotFoundError as e:
            logger.exception(f"Model file not found: {e}")
        except lgb.basic.LightGBMError as e:
            logger.exception(f"Error loading LightGBM model: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error loading model: {e}")

        meta_path = Path(path + ".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                obj._feature_names = meta.get("feature_names", [])
            except json.JSONDecodeError as e:
                logger.exception(f"Failed to decode metadata JSON: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error reading metadata: {e}")
        else:
            logger.warning(f"Metadata file {meta_path} does not exist")
        return obj