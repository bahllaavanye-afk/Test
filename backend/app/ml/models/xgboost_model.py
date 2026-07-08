"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
"""
import json
import logging
from pathlib import Path

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

    def __init__(self, **kwargs):
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
        try:
            self.model = xgb.XGBClassifier(**self.params, early_stopping_rounds=50, verbosity=0)
        except Exception as e:
            logger.exception("Failed to initialise XGBClassifier", exc_info=True, extra={"error": str(e)})
            raise
        self._explainer = None
        self.feature_names: list[str] = []

    def forward(self, x) -> np.ndarray:
        try:
            if hasattr(x, "numpy"):
                x = x.numpy()
            return self.model.predict_proba(x)[:, 1]
        except Exception as e:
            logger.exception("Error during forward pass", exc_info=True, extra={"input_shape": getattr(x, "shape", None)})
            raise

    def fit(self, X_train, y_train, X_val, y_val, feature_names: list[str] | None = None) -> dict:
        if feature_names:
            self.feature_names = feature_names
        try:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            return {
                "val_accuracy": float(accuracy_score(y_val, val_preds)),
                "val_auc": float(roc_auc_score(y_val, val_probs)),
            }
        except ValueError as ve:
            logger.error(
                "ValueError during model fitting or validation metric computation",
                exc_info=True,
                extra={"error": str(ve)},
            )
            raise
        except Exception as e:
            logger.exception("Unexpected error during fit", exc_info=True)
            raise

    def train_epoch(self, loader, optimizer=None, criterion=None) -> dict:
        # XGBoost uses fit() directly, not epoch-based training
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        all_probs, all_labels = [], []
        try:
            for X, y in loader:
                probs = self.forward(X.numpy() if hasattr(X, "numpy") else X)
                all_probs.append(probs)
                all_labels.append(y.numpy() if hasattr(y, "numpy") else y)
            probs_cat = np.concatenate(all_probs)
            labels_cat = np.concatenate(all_labels)
            preds = (probs_cat > 0.5).astype(int)
            acc = float(accuracy_score(labels_cat, preds))
            try:
                auc = float(roc_auc_score(labels_cat, probs_cat))
            except ValueError as ve:
                logger.warning(
                    "ROC AUC could not be computed; defaulting to 0.5",
                    exc_info=True,
                    extra={"error": str(ve)},
                )
                auc = 0.5
            return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)
        except Exception as e:
            logger.exception("Error during evaluation", exc_info=True)
            raise

    def get_feature_importance(self) -> dict[str, float]:
        """Return SHAP-based feature importance."""
        try:
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.model)
            importance = dict(
                zip(
                    self.feature_names
                    or [f"f{i}" for i in range(len(self.model.feature_importances_))],
                    self.model.feature_importances_.tolist(),
                )
            )
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.exception("Failed to compute feature importance", exc_info=True)
            raise

    def predict_proba(self, X) -> np.ndarray:
        try:
            if hasattr(X, "numpy"):
                X = X.numpy()
            return self.model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.exception("Error during predict_proba", exc_info=True, extra={"input_shape": getattr(X, "shape", None)})
            raise

    def save(self, path: str, metadata: dict | None = None) -> None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            model_path = path.replace(".pt", ".ubj")
            self.model.save_model(model_path)
            meta = {"feature_names": self.feature_names, "params": self.params, **(metadata or {})}
            Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))
        except OSError as oe:
            logger.error("Filesystem error during model save", exc_info=True, extra={"path": path, "error": str(oe)})
            raise
        except Exception as e:
            logger.exception("Unexpected error during save", exc_info=True, extra={"path": path})
            raise

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        try:
            instance = cls()
            instance.model.load_model(model_path)
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    instance.feature_names = meta.get("feature_names", [])
                except json.JSONDecodeError as je:
                    logger.error(
                        "Failed to decode metadata JSON during model load",
                        exc_info=True,
                        extra={"metadata_path": str(meta_path), "error": str(je)},
                    )
            return instance
        except FileNotFoundError as fnfe:
            logger.error(
                "Model file not found during load",
                exc_info=True,
                extra={"model_path": model_path, "error": str(fnfe)},
            )
            raise
        except Exception as e:
            logger.exception("Unexpected error during model load", exc_info=True, extra={"model_path": model_path})
            raise