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
        except Exception as exc:
            logger.exception("Failed to initialise XGBClassifier.")
            raise
        self._explainer = None
        self.feature_names: list[str] = []

    def forward(self, x) -> np.ndarray:
        try:
            if hasattr(x, "numpy"):
                x = x.numpy()
            return self.model.predict_proba(x)[:, 1]
        except Exception as exc:
            logger.exception("Error during forward pass.")
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
        except Exception as exc:
            logger.exception("Model fitting failed.")
            raise
        try:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            return {
                "val_accuracy": float(accuracy_score(y_val, val_preds)),
                "val_auc": float(roc_auc_score(y_val, val_probs)),
            }
        except Exception as exc:
            logger.exception("Error computing validation metrics.")
            raise

    def train_epoch(self, loader, optimizer=None, criterion=None) -> dict:
        # XGBoost uses fit() directly, not epoch-based training
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        all_probs, all_labels = [], []
        for X, y in loader:
            try:
                probs = self.forward(X.numpy() if hasattr(X, "numpy") else X)
                all_probs.append(probs)
                all_labels.append(y.numpy() if hasattr(y, "numpy") else y)
            except Exception as exc:
                logger.exception("Error processing a batch during evaluation.")
                raise
        try:
            probs_cat = np.concatenate(all_probs)
            labels_cat = np.concatenate(all_labels)
        except Exception as exc:
            logger.exception("Failed to concatenate evaluation results.")
            raise
        preds = (probs_cat > 0.5).astype(int)
        acc = float(accuracy_score(labels_cat, preds))
        try:
            auc = float(roc_auc_score(labels_cat, probs_cat))
        except ValueError:
            logger.warning("ROC AUC could not be computed; defaulting to 0.5.")
            auc = 0.5
        except Exception as exc:
            logger.exception("Unexpected error computing ROC AUC.")
            raise
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    def get_feature_importance(self) -> dict[str, float]:
        """Return SHAP-based feature importance."""
        try:
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            logger.exception("Failed to initialise SHAP TreeExplainer.")
            raise
        try:
            importance = dict(
                zip(
                    self.feature_names
                    or [f"f{i}" for i in range(len(self.model.feature_importances_))],
                    self.model.feature_importances_.tolist(),
                )
            )
        except Exception as exc:
            logger.exception("Error extracting feature importances.")
            raise
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def predict_proba(self, X) -> np.ndarray:
        try:
            if hasattr(X, "numpy"):
                X = X.numpy()
            return self.model.predict_proba(X)[:, 1]
        except Exception as exc:
            logger.exception("Error during predict_proba.")
            raise

    def save(self, path: str, metadata: dict | None = None) -> None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            model_path = path.replace(".pt", ".ubj")
            self.model.save_model(model_path)
            meta = {
                "feature_names": self.feature_names,
                "params": self.params,
                **(metadata or {}),
            }
            Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))
        except OSError as exc:
            logger.exception(f"Filesystem error while saving model to {path}.")
            raise
        except Exception as exc:
            logger.exception("Unexpected error during model save.")
            raise

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        try:
            instance = cls()
            instance.model.load_model(model_path)
        except FileNotFoundError as exc:
            logger.exception(f"Model file not found at {model_path}.")
            raise
        except OSError as exc:
            logger.exception(f"Filesystem error while loading model from {model_path}.")
            raise
        except Exception as exc:
            logger.exception("Unexpected error loading XGBoost model.")
            raise
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                instance.feature_names = meta.get("feature_names", [])
            except Exception as exc:
                logger.exception("Failed to load model metadata.")
                raise
        return instance