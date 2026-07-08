"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
"""
import json
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import shap
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGB_AVAILABLE = False


class XGBoostClassifier(AbstractModel):
    """XGBoost binary classifier with optional signal filtering.

    The class provides methods to generate entry/exit signals with tighter
    conditions and confirmation filters. Default thresholds can be overridden
    at construction time.
    """

    model_type = "xgboost"

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        entry_threshold: float = 0.55,
        exit_threshold: float = 0.45,
        confirmation_window: int = 5,
        **kwargs,
    ) -> None:
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }
        # Early stopping is handled via the fit method; verbosity suppressed.
        self.model = xgb.XGBClassifier(**self.params, early_stopping_rounds=50, verbosity=0)
        self._explainer = None
        self.feature_names: List[str] = []

        # Signal filtering parameters
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.confirmation_window = max(1, confirmation_window)

        # Internal state for rolling confirmation (used when calling generate_signal sequentially)
        self._recent_probs: Deque[float] = deque(maxlen=self.confirmation_window)

    # --------------------------------------------------------------------- #
    # Core model interface
    # --------------------------------------------------------------------- #
    def forward(self, x) -> np.ndarray:
        """Return the probability of the positive class."""
        if hasattr(x, "numpy"):
            x = x.numpy()
        return self.model.predict_proba(x)[:, 1]

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names: List[str] | None = None,
    ) -> dict:
        """Fit the model and return validation metrics."""
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

    def train_epoch(self, loader, optimizer=None, criterion=None) -> dict:
        # XGBoost uses fit() directly, not epoch-based training
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        """Evaluate on a data loader and return standard metrics."""
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
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    # --------------------------------------------------------------------- #
    # Feature importance utilities
    # --------------------------------------------------------------------- #
    def get_feature_importance(self) -> dict[str, float]:
        """Return SHAP-based feature importance sorted descending."""
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

    # --------------------------------------------------------------------- #
    # Prediction helpers
    # --------------------------------------------------------------------- #
    def predict_proba(self, X) -> np.ndarray:
        """Convenience wrapper for forward."""
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    # --------------------------------------------------------------------- #
    # Signal generation
    # --------------------------------------------------------------------- #
    def _update_recent_probs(self, prob: float) -> None:
        """Maintain a rolling window of recent probabilities."""
        self._recent_probs.append(prob)

    def _confirmation_passed(self) -> bool:
        """Check if the rolling average exceeds the entry threshold."""
        if len(self._recent_probs) < self.confirmation_window:
            return False
        return float(np.mean(self._recent_probs)) >= self.entry_threshold

    def generate_signal(
        self,
        X,
        *,
        use_confirmation: bool = True,
        raw_prob: float | None = None,
    ) -> Tuple[int, float]:
        """Generate a trading signal for a single observation.

        Returns:
            signal (int): 1 for entry/long, -1 for exit/short, 0 for no action.
            prob (float): The raw probability of the positive class.
        """
        prob = raw_prob if raw_prob is not None else self.forward(X)[0]
        # Update rolling window regardless of confirmation usage
        if use_confirmation:
            self._update_recent_probs(prob)

        # Entry logic: tighten condition with higher threshold and optional confirmation
        if prob >= self.entry_threshold:
            if not use_confirmation or self._confirmation_passed():
                return 1, prob

        # Exit logic: loosen condition but require dropping below exit threshold
        if prob <= self.exit_threshold:
            # Reset recent probs to avoid stale confirmation after exit
            self._recent_probs.clear()
            return -1, prob

        return 0, prob

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save(self, path: str, metadata: dict | None = None) -> None:
        """Save model parameters and metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model_path = path.replace(".pt", ".ubj")
        self.model.save_model(model_path)
        meta = {
            "feature_names": self.feature_names,
            "params": self.params,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "confirmation_window": self.confirmation_window,
            **(metadata or {}),
        }
        Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        """Load a model from disk, restoring hyper‑parameters and metadata."""
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        instance.model.load_model(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance.feature_names = meta.get("feature_names", [])
            instance.entry_threshold = meta.get("entry_threshold", 0.55)
            instance.exit_threshold = meta.get("exit_threshold", 0.45)
            instance.confirmation_window = meta.get("confirmation_window", 5)
        return instance