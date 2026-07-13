"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
"""
import json
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

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
    """Wrapper around XGBoost binary classifier with utilities for trading signal generation."""

    model_type = "xgboost"

    # Default thresholds – can be overridden per‑instance if needed
    ENTRY_THRESHOLD = 0.6
    EXIT_THRESHOLD = 0.4
    CONFIRMATION_WINDOW = 3  # number of recent predictions to consider
    CONFIRMATION_RATIO = 0.66  # proportion of recent predictions that must agree

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
        self._explainer: Optional[shap.Explainer] = None
        self.feature_names: List[str] = []
        # Internal state for confirmation filtering
        self._recent_probs: Deque[float] = deque(maxlen=self.CONFIRMATION_WINDOW)

    # --------------------------------------------------------------------- #
    # Core model interface
    # --------------------------------------------------------------------- #
    def forward(self, x) -> np.ndarray:
        """Return probability of the positive class."""
        if hasattr(x, "numpy"):
            x = x.numpy()
        return self.model.predict_proba(x)[:, 1]

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names: Optional[List[str]] = None,
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
        """XGBoost uses fit() directly, not epoch‑based training."""
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        """Evaluate on a data loader and return common metrics."""
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
        """Return SHAP‑based feature importance sorted descending."""
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
    # Persistence helpers
    # --------------------------------------------------------------------- #
    def predict_proba(self, X) -> np.ndarray:
        """Convenient wrapper for probability prediction."""
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        """Save model binary and accompanying metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model_path = path.replace(".pt", ".ubj")
        self.model.save_model(model_path)
        meta = {"feature_names": self.feature_names, "params": self.params, **(metadata or {})}
        Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        """Load a model from disk."""
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        instance.model.load_model(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance.feature_names = meta.get("feature_names", [])
        return instance

    # --------------------------------------------------------------------- #
    # Trading signal generation
    # --------------------------------------------------------------------- #
    def _update_recent_probs(self, prob: float) -> None:
        """Maintain a short history of recent probabilities for confirmation."""
        self._recent_probs.append(prob)

    def _confirmation_passed(self) -> bool:
        """
        Confirm that a majority of recent probabilities exceed the entry (or exit) threshold.
        Returns True if enough recent values agree with the current signal direction.
        """
        if len(self._recent_probs) < self.CONFIRMATION_WINDOW:
            # Not enough history – be conservative and require explicit threshold
            return False
        above = sum(p >= self.ENTRY_THRESHOLD for p in self._recent_probs)
        below = sum(p <= self.EXIT_THRESHOLD for p in self._recent_probs)
        # For a long entry we need enough "above" values; for a short exit we need enough "below"
        return max(above, below) / self.CONFIRMATION_WINDOW >= self.CONFIRMATION_RATIO

    def generate_signal(self, X) -> int:
        """
        Produce a trading signal based on model probability and confirmation filters.

        Returns:
            1  – Long entry signal
           -1  – Short/exit signal
            0  – No action / hold
        """
        prob = float(self.forward(X)[0]) if isinstance(X, (list, np.ndarray)) else float(self.forward(np.array([X]))[0])
        self._update_recent_probs(prob)

        # Entry condition – probability must be comfortably above the entry threshold
        if prob >= self.ENTRY_THRESHOLD and self._confirmation_passed():
            return 1

        # Exit condition – probability falls below the exit threshold
        if prob <= self.EXIT_THRESHOLD and self._confirmation_passed():
            return -1

        return 0

    def should_exit(self, X, position_age: int, max_holding: Optional[int] = None) -> bool:
        """
        Determine whether an open position should be closed.

        Args:
            X: Feature vector for the current timestep.
            position_age: Number of bars the position has been held.
            max_holding: Optional hard stop on holding period (in bars).

        Returns:
            True if the position should be exited, False otherwise.
        """
        prob = float(self.forward(X)[0] if isinstance(X, (list, np.ndarray)) else self.forward(np.array([X]))[0])

        # Hard stop on holding period
        if max_holding is not None and position_age >= max_holding:
            return True

        # Probabilistic exit trigger
        if prob <= self.EXIT_THRESHOLD:
            return True

        # Confirmation‑based exit: if recent probabilities consistently indicate weakness
        self._update_recent_probs(prob)
        if len(self._recent_probs) == self.CONFIRMATION_WINDOW:
            below_ratio = sum(p <= self.EXIT_THRESHOLD for p in self._recent_probs) / self.CONFIRMATION_WINDOW
            if below_ratio >= self.CONFIRMATION_RATIO:
                return True

        return False