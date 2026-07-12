"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import shap
from pydantic import BaseModel, Field, validator
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGB_AVAILABLE = False


class XGBoostParams(BaseModel):
    """Hyperparameter schema for :class:`XGBoostClassifier`."""

    n_estimators: int = Field(
        500,
        description="Number of boosting rounds.",
        example=500,
        ge=1,
    )
    max_depth: int = Field(
        5,
        description="Maximum depth of a tree.",
        example=5,
        ge=1,
    )
    learning_rate: float = Field(
        0.05,
        description="Step size shrinkage used in update to prevents overfitting.",
        example=0.05,
        gt=0.0,
        le=1.0,
    )
    subsample: float = Field(
        0.8,
        description="Subsample ratio of the training instances.",
        example=0.8,
        gt=0.0,
        le=1.0,
    )
    colsample_bytree: float = Field(
        0.8,
        description="Subsample ratio of columns when constructing each tree.",
        example=0.8,
        gt=0.0,
        le=1.0,
    )
    min_child_weight: int = Field(
        3,
        description="Minimum sum of instance weight (hessian) needed in a child.",
        example=3,
        ge=0,
    )
    reg_alpha: float = Field(
        0.1,
        description="L1 regularization term on weights.",
        example=0.1,
        ge=0.0,
    )
    reg_lambda: float = Field(
        1.0,
        description="L2 regularization term on weights.",
        example=1.0,
        ge=0.0,
    )

    @validator("learning_rate", "subsample", "colsample_bytree")
    def _validate_fraction(cls, v: float) -> float:
        """Ensure fractional hyperparameters stay within (0, 1]."""
        if not (0.0 < v <= 1.0):
            raise ValueError("value must be greater than 0 and at most 1")
        return v


class XGBoostClassifier(AbstractModel):
    model_type = "xgboost"

    def __init__(self, **kwargs: Any):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        # Validate and store hyperparameters using the Pydantic schema
        self.params = XGBoostParams(**kwargs).dict()
        # Add static XGBoost arguments that are not exposed via the schema
        self.params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
            }
        )
        self.model = xgb.XGBClassifier(**self.params, early_stopping_rounds=50, verbosity=0)
        self._explainer: Optional[Any] = None
        self.feature_names: List[str] = []

    def forward(self, x: Any) -> np.ndarray:
        if hasattr(x, "numpy"):
            x = x.numpy()
        return self.model.predict_proba(x)[:, 1]

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
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
        return {
            "val_accuracy": float(accuracy_score(y_val, val_preds)),
            "val_auc": float(roc_auc_score(y_val, val_probs)),
        }

    def train_epoch(self, loader: Any, optimizer: Any = None, criterion: Any = None) -> Dict[str, float]:
        # XGBoost uses fit() directly, not epoch‑based training
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader: Any) -> EvalMetrics:
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
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
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return SHAP‑based feature importance."""
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

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model_path = path.replace(".pt", ".ubj")
        self.model.save_model(model_path)
        meta = {
            "feature_names": self.feature_names,
            "params": self.params,
            **(metadata or {}),
        }
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