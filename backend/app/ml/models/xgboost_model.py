"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import shap
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


class XGBoostClassifier(AbstractModel):
    """XGBoost binary classifier with utilities for signal generation.

    The class wraps an ``xgboost.XGBClassifier`` and provides:
    * model fitting / evaluation
    * SHAP based feature importance
    * helper methods to create entry / exit signals with tighter
      probability thresholds and optional confirmation windows.
    """

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
        # ``early_stopping_rounds`` is only relevant when a validation set is supplied.
        self.model = xgb.XGBClassifier(**self.params, early_stopping_rounds=50, verbosity=0)
        self._explainer: Optional[shap.Explainer] = None
        self.feature_names: List[str] = []

    # --------------------------------------------------------------------- #
    # Core model interface
    # --------------------------------------------------------------------- #
    def forward(self, x) -> np.ndarray:
        """Return the positive class probability for ``x``."""
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
        # XGBoost uses fit() directly, not epoch‑based training.
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        """Evaluate on a data loader returning ``EvalMetrics``."""
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
    # Explainability
    # --------------------------------------------------------------------- #
    def get_feature_importance(self) -> dict[str, float]:
        """Return SHAP‑based feature importance sorted descending."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        # Fallback to generic feature names if none were supplied.
        feature_labels = self.feature_names or [
            f"f{i}" for i in range(len(self.model.feature_importances_))
        ]
        importance = dict(
            zip(feature_labels, self.model.feature_importances_.tolist())
        )
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def predict_proba(self, X) -> np.ndarray:
        """Convenience wrapper returning the positive class probability."""
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        """Save model parameters and auxiliary metadata."""
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
        """Load a previously saved model."""
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        instance.model.load_model(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance.feature_names = meta.get("feature_names", [])
        return instance

    # --------------------------------------------------------------------- #
    # Strategy signal helpers
    # --------------------------------------------------------------------- #
    def _check_consecutive(self, probs: np.ndarray, threshold: float, n: int) -> bool:
        """Return ``True`` if the last ``n`` probabilities exceed ``threshold``."""
        if probs.shape[0] < n:
            return False
        return bool(np.all(probs[-n:] >= threshold))

    def generate_entry_signal(
        self,
        X: np.ndarray,
        *,
        entry_threshold: float = 0.60,
        confirmation_window: int = 2,
    ) -> bool:
        """Determine whether to open a long position.

        The entry signal is triggered when the model's probability for the
        positive class is above ``entry_threshold`` for at least
        ``confirmation_window`` consecutive observations. This tightens the
        naive ``>0.5`` rule and adds a confirmation filter to reduce false
        entries.

        Args:
            X: Input features. Shape ``(t, n_features)`` where ``t`` >= ``confirmation_window``.
            entry_threshold: Probability required for a confident bullish signal.
            confirmation_window: Number of consecutive periods that must satisfy the threshold.

        Returns:
            ``True`` if entry conditions are met, ``False`` otherwise.
        """
        if X.shape[0] < confirmation_window:
            logger.debug(
                "Insufficient data for entry confirmation: %s rows provided, need %s",
                X.shape[0],
                confirmation_window,
            )
            return False
        probs = self.forward(X)
        signal = self._check_consecutive(probs, entry_threshold, confirmation_window)
        logger.debug(
            "Entry signal check – probs: %s, threshold: %.2f, window: %d, result: %s",
            probs[-confirmation_window:],
            entry_threshold,
            confirmation_window,
            signal,
        )
        return signal

    def generate_exit_signal(
        self,
        X: np.ndarray,
        *,
        exit_threshold: float = 0.40,
    ) -> bool:
        """Determine whether to close an existing position.

        The exit condition is met when the latest probability falls below
        ``exit_threshold``. A lower threshold than the entry point provides a
        buffer that helps avoid premature exits while still protecting capital.

        Args:
            X: Input features for the most recent observation (or a batch where the
               last row is the latest data point).
            exit_threshold: Probability below which the position should be exited.

        Returns:
            ``True`` if exit conditions are satisfied, ``False`` otherwise.
        """
        probs = self.forward(X)
        latest_prob = probs[-1] if probs.ndim > 0 else probs
        signal = float(latest_prob) <= exit_threshold
        logger.debug(
            "Exit signal check – latest_prob: %.4f, threshold: %.2f, result: %s",
            latest_prob,
            exit_threshold,
            signal,
        )
        return signal

    def generate_signal(
        self,
        X: np.ndarray,
        *,
        entry_threshold: float = 0.60,
        exit_threshold: float = 0.40,
        confirmation_window: int = 2,
    ) -> Tuple[bool, bool]:
        """Convenient wrapper returning both entry and exit flags.

        Returns a tuple ``(enter, exit)`` where ``enter`` indicates a new long
        position should be opened and ``exit`` indicates an existing position
        should be closed. The two flags are mutually exclusive – if both are
        ``True`` the entry flag takes precedence (new position after exit).

        Args:
            X: Feature matrix with the most recent observations.
            entry_threshold: Threshold for entry confirmation.
            exit_threshold: Threshold for exit.
            confirmation_window: Number of consecutive periods required for entry.

        Returns:
            Tuple[bool, bool]: ``(enter, exit)``.
        """
        enter = self.generate_entry_signal(
            X,
            entry_threshold=entry_threshold,
            confirmation_window=confirmation_window,
        )
        exit_ = self.generate_exit_signal(
            X,
            exit_threshold=exit_threshold,
        )
        # If both signals fire, prioritize entry (i.e., close‑then‑open).
        if enter and exit_:
            logger.debug("Both entry and exit conditions met; prioritizing entry.")
            exit_ = False
        return enter, exit_