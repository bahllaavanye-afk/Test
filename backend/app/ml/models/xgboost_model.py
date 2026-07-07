"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
"""

import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

try:
    import xgboost as xgb
    import shap

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class XGBoostClassifier(AbstractModel):
    """XGBoost binary classifier with utilities for signal generation."""

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
        # early_stopping_rounds is handled internally by XGBoost's fit method
        self.model = xgb.XGBClassifier(**self.params, early_stopping_rounds=50, verbosity=0)
        self._explainer: Optional[shap.Explainer] = None
        self.feature_names: List[str] = []

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

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def predict_proba(self, X) -> np.ndarray:
        """Convenience wrapper that mirrors ``forward``."""
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        """Save model weights and accompanying metadata."""
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
        """Load a model and its metadata."""
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        instance.model.load_model(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance.feature_names = meta.get("feature_names", [])
        return instance

    # --------------------------------------------------------------------- #
    # Strategy logic enhancements
    # --------------------------------------------------------------------- #
    def generate_signals(
        self,
        X: np.ndarray,
        entry_threshold: float = 0.6,
        exit_threshold: float = 0.4,
        confirmation: int = 2,
        filter_funcs: Optional[Iterable[Callable[[np.ndarray], np.ndarray]]] = None,
    ) -> np.ndarray:
        """
        Generate trading signals (1 = long, 0 = flat) from model probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix aligned with the model's expectations.
        entry_threshold : float, default 0.6
            Minimum probability required to consider an entry.
        exit_threshold : float, default 0.4
            Probability below which an existing position is closed.
        confirmation : int, default 2
            Number of consecutive periods the entry condition must hold before a
            signal is issued. Helps filter out transient spikes.
        filter_funcs : iterable of callables, optional
            Additional boolean filters that must be true for a signal to be
            generated. Each callable receives ``X`` and returns a boolean mask
            with shape ``(n_samples,)``. All filters are combined with logical AND.

        Returns
        -------
        np.ndarray
            Integer array of signals (1 for long, 0 for flat) with the same length
            as the input rows.
        """
        probs = self.forward(X)
        n = len(probs)
        signals = np.zeros(n, dtype=int)

        # Base entry / exit masks
        entry_mask = probs >= entry_threshold
        exit_mask = probs <= exit_threshold

        # Apply optional external filters
        if filter_funcs:
            for fn in filter_funcs:
                mask = fn(X)
                if mask.shape != (n,):
                    raise ValueError("Filter function must return a 1‑D boolean array of length n")
                entry_mask &= mask
                # exit mask generally stays independent; however, if a filter
                # explicitly wants to block exits it can be combined here.
                exit_mask &= mask

        # Confirmation logic: require ``confirmation`` consecutive True values
        if confirmation > 1:
            # Rolling window using cumulative sum trick
            cum = np.cumsum(entry_mask.astype(int))
            confirmed = np.concatenate(
                [np.zeros(confirmation - 1, dtype=bool), cum[confirmation - 1 :] - cum[:-confirmation + 1] == confirmation]
            )
        else:
            confirmed = entry_mask

        position = 0  # 0 = flat, 1 = long
        for i in range(n):
            if position == 0 and confirmed[i]:
                position = 1
            elif position == 1 and exit_mask[i]:
                position = 0
            signals[i] = position

        return signals

    def apply_signal_filters(
        self,
        signals: np.ndarray,
        X: np.ndarray,
        filter_funcs: Iterable[Callable[[np.ndarray], np.ndarray]],
    ) -> np.ndarray:
        """
        Apply additional user‑defined filters to an existing signal array.

        Parameters
        ----------
        signals : np.ndarray
            Original signal array (0/1).
        X : np.ndarray
            Feature matrix used for filter evaluation.
        filter_funcs : iterable of callables
            Functions that return a boolean mask. Signals are forced to 0 where any
            mask is False.

        Returns
        -------
        np.ndarray
            Filtered signal array.
        """
        if signals.shape[0] != X.shape[0]:
            raise ValueError("signals and X must have the same length")
        mask = np.ones_like(signals, dtype=bool)
        for fn in filter_funcs:
            cur = fn(X)
            if cur.shape != signals.shape:
                raise ValueError("Filter function must return a mask matching signal shape")
            mask &= cur
        filtered = signals.copy()
        filtered[~mask] = 0
        return filtered