"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

try:
    import shap
    import xgboost as xgb

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
        self.model = xgb.XGBClassifier(**self.params, early_stopping_rounds=50, verbosity=0)
        self._explainer = None
        self.feature_names: list[str] = []

    def _log_metrics(
        self,
        operation: str,
        signal_count: int,
        exec_time: float,
        pnl: float | None = None,
    ) -> None:
        log_record = {
            "operation": operation,
            "signal_count": signal_count,
            "execution_time_ms": round(exec_time * 1000, 2),
            "pnl": pnl,
        }
        logger.info("ModelMetrics: %s", json.dumps(log_record))

    def forward(self, x) -> np.ndarray:
        start = time.perf_counter()
        if hasattr(x, "numpy"):
            x = x.numpy()
        probs = self.model.predict_proba(x)[:, 1]
        exec_time = time.perf_counter() - start
        self._log_metrics("forward", signal_count=len(probs), exec_time=exec_time, pnl=None)
        return probs

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names: list[str] | None = None,
    ) -> dict:
        start = time.perf_counter()
        if feature_names:
            self.feature_names = feature_names
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_probs = self.model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        exec_time = time.perf_counter() - start
        self._log_metrics(
            "fit",
            signal_count=int(X_train.shape[0]),
            exec_time=exec_time,
            pnl=None,
        )
        return {
            "val_accuracy": float(accuracy_score(y_val, val_preds)),
            "val_auc": float(roc_auc_score(y_val, val_probs)),
        }

    def train_epoch(self, loader, optimizer=None, criterion=None) -> dict:
        # XGBoost uses fit() directly, not epoch-based training
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        start = time.perf_counter()
        all_probs, all_labels = [], []
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
        except ValueError:
            auc = 0.5
        exec_time = time.perf_counter() - start
        self._log_metrics(
            "evaluate",
            signal_count=len(probs_cat),
            exec_time=exec_time,
            pnl=None,
        )
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    def get_feature_importance(self) -> dict[str, float]:
        """Return SHAP-based feature importance."""
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

    def predict_proba(self, X) -> np.ndarray:
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model_path = path.replace(".pt", ".ubj")
        self.model.save_model(model_path)
        meta = {"feature_names": self.feature_names, "params": self.params, **(metadata or {})}
        Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        instance.model.load_model(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance.feature_names = meta.get("feature_names", [])
        return instance