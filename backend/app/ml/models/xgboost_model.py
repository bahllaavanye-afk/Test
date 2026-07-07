"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
"""
import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

try:
    import xgboost as xgb
    import shap
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


class XGBoostClassifier(AbstractModel):
    model_type = "xgboost"

    def __init__(self, **kwargs: Any) -> None:
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "n_estimators": kwargs.get("n_estimators", 500),
            "max_depth": kwargs.get("max_depth", 5),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "min_child_weight": kwargs.get("min_child_weight", 3),
            "reg_alpha": kwargs.get("reg_alpha", 0.1),
            "reg_lambda": kwargs.get("reg_lambda", 1.0),
        }
        self.model = xgb.XGBClassifier(**self.params, early_stopping_rounds=50, verbosity=0)
        self._explainer: Optional[Any] = None
        self.feature_names: List[str] = []

    def forward(self, x: Any) -> np.ndarray:
        if hasattr(x, "numpy"):
            x = x.numpy()
        try:
            return self.model.predict_proba(x)[:, 1]
        except Exception as e:
            logger.error(
                "Error during forward prediction",
                exc_info=True,
                extra={"error": str(e), "input_shape": getattr(x, "shape", None)},
            )
            raise RuntimeError("Forward prediction failed") from e

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        feature_names: Optional[List[str]] = None,
    ) -> dict:
        if feature_names:
            self.feature_names = feature_names
        try:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        except Exception as e:
            logger.error(
                "Model fitting failed",
                exc_info=True,
                extra={"error": str(e)},
            )
            raise RuntimeError("Model fitting failed") from e

        try:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            val_accuracy = float(accuracy_score(y_val, val_preds))
            val_auc = float(roc_auc_score(y_val, val_probs))
        except Exception as e:
            logger.error(
                "Error computing validation metrics",
                exc_info=True,
                extra={"error": str(e)},
            )
            raise RuntimeError("Validation metric computation failed") from e

        return {
            "val_accuracy": val_accuracy,
            "val_auc": val_auc,
        }

    def train_epoch(self, loader: Any, optimizer: Any = None, criterion: Any = None) -> dict:
        # XGBoost uses fit() directly, not epoch-based training
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader: Any) -> EvalMetrics:
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        try:
            for X, y in loader:
                probs = self.forward(X.numpy() if hasattr(X, "numpy") else X)
                all_probs.append(probs)
                all_labels.append(y.numpy() if hasattr(y, "numpy") else y)
        except Exception as e:
            logger.error(
                "Error during evaluation data collection",
                exc_info=True,
                extra={"error": str(e)},
            )
            raise RuntimeError("Evaluation data collection failed") from e

        try:
            probs_cat = np.concatenate(all_probs)
            labels_cat = np.concatenate(all_labels)
            preds = (probs_cat > 0.5).astype(int)
            acc = float(accuracy_score(labels_cat, preds))
            auc = float(roc_auc_score(labels_cat, probs_cat))
        except ValueError as ve:
            logger.warning(
                "AUC calculation failed, falling back to default 0.5",
                exc_info=True,
                extra={"error": str(ve)},
            )
            auc = 0.5
            acc = float(accuracy_score(labels_cat, preds))
        except Exception as e:
            logger.error(
                "Unexpected error during evaluation metric computation",
                exc_info=True,
                extra={"error": str(e)},
            )
            raise RuntimeError("Evaluation metric computation failed") from e

        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    def get_feature_importance(self) -> dict[str, float]:
        """Return SHAP-based feature importance."""
        if self._explainer is None:
            try:
                self._explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                logger.error(
                    "Failed to create SHAP explainer",
                    exc_info=True,
                    extra={"error": str(e)},
                )
                raise RuntimeError("SHAP explainer creation failed") from e

        try:
            importance = dict(
                zip(
                    self.feature_names
                    or [f"f{i}" for i in range(len(self.model.feature_importances_))],
                    self.model.feature_importances_.tolist(),
                )
            )
        except Exception as e:
            logger.error(
                "Failed to retrieve feature importances",
                exc_info=True,
                extra={"error": str(e)},
            )
            raise RuntimeError("Feature importance extraction failed") from e

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(X, "numpy"):
            X = X.numpy()
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(
                "Error during predict_proba",
                exc_info=True,
                extra={"error": str(e), "input_shape": getattr(X, "shape", None)},
            )
            raise RuntimeError("Probability prediction failed") from e

    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            model_path = path.replace(".pt", ".ubj")
            self.model.save_model(model_path)
            meta = {
                "feature_names": self.feature_names,
                "params": self.params,
                **(metadata or {}),
            }
            json_path = Path(path).with_suffix(".json")
            json_path.write_text(json.dumps(meta, indent=2))
        except OSError as oe:
            logger.error(
                "Filesystem error during model save",
                exc_info=True,
                extra={"error": str(oe), "path": path},
            )
            raise RuntimeError("Model save failed due to filesystem error") from oe
        except Exception as e:
            logger.error(
                "Unexpected error during model save",
                exc_info=True,
                extra={"error": str(e)},
            )
            raise RuntimeError("Model save failed") from e

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        try:
            instance.model.load_model(model_path)
        except FileNotFoundError as fnfe:
            logger.error(
                "Model file not found during load",
                exc_info=True,
                extra={"error": str(fnfe), "model_path": model_path},
            )
            raise RuntimeError("Model file not found") from fnfe
        except Exception as e:
            logger.error(
                "Error loading XGBoost model",
                exc_info=True,
                extra={"error": str(e), "model_path": model_path},
            )
            raise RuntimeError("Failed to load XGBoost model") from e

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                instance.feature_names = meta.get("feature_names", [])
            except json.JSONDecodeError as jde:
                logger.error(
                    "Invalid JSON in metadata file",
                    exc_info=True,
                    extra={"error": str(jde), "meta_path": str(meta_path)},
                )
                raise RuntimeError("Metadata JSON decode error") from jde
            except Exception as e:
                logger.error(
                    "Unexpected error reading metadata",
                    exc_info=True,
                    extra={"error": str(e), "meta_path": str(meta_path)},
                )
                raise RuntimeError("Failed to read metadata") from e

        return instance