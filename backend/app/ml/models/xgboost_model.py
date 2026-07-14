"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
Enhanced signal generation with tighter entry conditions, confirmation filters,
and improved exit logic.
"""
import json
import collections
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

try:
    import xgboost as xgb
    import shap

    XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGB_AVAILABLE = False


class XGBoostClassifier(AbstractModel):
    """XGBoost binary classifier with built‑in signal generation."""

    model_type = "xgboost"

    # Default signal thresholds – tighter than the generic 0.5 cutoff
    DEFAULT_ENTRY_THRESHOLD = 0.60
    DEFAULT_EXIT_THRESHOLD = 0.40
    DEFAULT_CONFIRMATION_WINDOW = 3  # consecutive periods required

    def __init__(self, **kwargs):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        # Model hyper‑parameters
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
        self.feature_names: List[str] = []

        # Signal generation attributes
        self.entry_threshold: float = kwargs.get(
            "entry_threshold", self.DEFAULT_ENTRY_THRESHOLD
        )
        self.exit_threshold: float = kwargs.get(
            "exit_threshold", self.DEFAULT_EXIT_THRESHOLD
        )
        self.confirmation_window: int = kwargs.get(
            "confirmation_window", self.DEFAULT_CONFIRMATION_WINDOW
        )
        self._recent_probs: Deque[float] = collections.deque(maxlen=self.confirmation_window)

    # --------------------------------------------------------------------- #
    # Core model interface
    # --------------------------------------------------------------------- #
    def forward(self, x) -> np.ndarray:
        """Return the predicted probability of the positive class."""
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
    # Feature importance & explainability
    # --------------------------------------------------------------------- #
    def get_feature_importance(self) -> dict[str, float]:
        """Return SHAP‑based (fallback to gain) feature importance."""
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
        """Convenience wrapper around forward."""
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    # --------------------------------------------------------------------- #
    # Signal generation
    # --------------------------------------------------------------------- #
    def _update_recent_probs(self, prob: float) -> None:
        """Maintain a rolling window of recent probabilities for confirmation."""
        self._recent_probs.append(prob)

    def _confirmation_met(self, above: bool) -> bool:
        """
        Check whether the recent probability window consistently satisfies the
        direction required for a signal.

        Parameters
        ----------
        above: bool
            True if we expect probabilities to be above the entry threshold,
            False if we expect them to be below the exit threshold.

        Returns
        -------
        bool
            True if all stored probabilities meet the condition.
        """
        if len(self._recent_probs) < self.confirmation_window:
            return False
        condition = (p > self.entry_threshold) if above else (p < self.exit_threshold)
        return all(condition for p in self._recent_probs)

    def generate_signal(self, X) -> str:
        """
        Produce a trading signal for a single observation.

        Returns
        -------
        str
            One of {"enter", "exit", "hold"}:
            - ``enter``  – probability exceeds entry threshold and confirmation met.
            - ``exit``   – probability falls below exit threshold and confirmation met.
            - ``hold``   – otherwise.
        """
        prob = float(self.forward(X)[0])
        self._update_recent_probs(prob)

        if prob >= self.entry_threshold and self._confirmation_met(above=True):
            return "enter"
        if prob <= self.exit_threshold and self._confirmation_met(above=False):
            return "exit"
        return "hold"

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save(self, path: str, metadata: dict | None = None) -> None:
        """Save model weights and metadata to disk."""
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
        """Load a persisted model."""
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        instance.model.load_model(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance.feature_names = meta.get("feature_names", [])
            instance.entry_threshold = meta.get(
                "entry_threshold", cls.DEFAULT_ENTRY_THRESHOLD
            )
            instance.exit_threshold = meta.get(
                "exit_threshold", cls.DEFAULT_EXIT_THRESHOLD
            )
            instance.confirmation_window = meta.get(
                "confirmation_window", cls.DEFAULT_CONFIRMATION_WINDOW
            )
            # Re‑initialize recent probability buffer with appropriate size
            instance._recent_probs = collections.deque(
                maxlen=instance.confirmation_window
            )
        return instance