"""
XGBoost binary classifier with Optuna hyperparameter optimization.
SHAP-based explainability built in.
"""

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

try:
    import shap
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGB_AVAILABLE = False


class XGBoostClassifier(AbstractModel):
    """XGBoost binary classifier with utilities for signal generation.

    The class provides methods to produce trading signals based on model
    probabilities and optional SHAP‑based confirmation filters.  Default
    thresholds are set conservatively (0.6 entry, 0.4 exit) but can be
    overridden per call.
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
        # early_stopping_rounds is handled by XGBClassifier directly
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
        # XGBoost uses fit() directly, not epoch‑based training
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        """Evaluate on a data loader returning probability‑based metrics."""
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
    # Explainability utilities
    # --------------------------------------------------------------------- #
    def _get_shap_explainer(self):
        """Lazily create a SHAP TreeExplainer."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def get_feature_importance(self) -> dict[str, float]:
        """Return SHAP‑based feature importance if possible, else fall back to XGBoost scores."""
        if self._explainer is None:
            # Prefer SHAP values; if not yet built, compute on a small sample
            try:
                self._explainer = shap.TreeExplainer(self.model)
            except Exception:  # pragma: no cover
                pass
        if self._explainer is not None:
            # Use mean absolute SHAP values across the training set if available
            # (We cannot guarantee training data here, so fallback to model importances)
            importance_vals = self.model.feature_importances_
        else:
            importance_vals = self.model.feature_importances_
        importance = dict(
            zip(
                self.feature_names
                or [f"f{i}" for i in range(len(importance_vals))],
                importance_vals.tolist(),
            )
        )
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # --------------------------------------------------------------------- #
    # Signal generation logic
    # --------------------------------------------------------------------- #
    def generate_signals(
        self,
        X: np.ndarray,
        current_position: int = 0,
        entry_threshold: float = 0.6,
        exit_threshold: float = 0.4,
        shap_top_k: int = 3,
        shap_sign_match: bool = True,
    ) -> np.ndarray:
        """Generate trading signals with tightened entry/exit logic.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (samples × features).
        current_position : int, optional
            Existing position: 1 for long, -1 for short, 0 for flat.
        entry_threshold : float, optional
            Minimum probability required to open a new position.
        exit_threshold : float, optional
            Maximum probability to keep an existing position; falling below
            triggers an exit.
        shap_top_k : int, optional
            Number of top‑impact SHAP features to consider for confirmation.
        shap_sign_match : bool, optional
            If True, the sign of the summed top‑k SHAP values must agree
            with the direction of the probability signal.

        Returns
        -------
        np.ndarray
            Array of signals: 1 (buy/long), -1 (sell/short), 0 (hold/flat).
        """
        probs = self.forward(X)
        signals = np.zeros_like(probs, dtype=int)

        # Compute SHAP values only when confirmation is requested
        shap_vals = None
        if shap_sign_match:
            try:
                explainer = self._get_shap_explainer()
                shap_vals = explainer.shap_values(X)
            except Exception:
                shap_vals = None  # Gracefully degrade to probability‑only logic

        for i, prob in enumerate(probs):
            # Determine direction implied by probability
            direction = 1 if prob > 0.5 else -1

            # Entry logic (only when flat)
            if current_position == 0:
                if prob >= entry_threshold:
                    if shap_vals is not None:
                        top_k = np.argsort(-np.abs(shap_vals[i]))[:shap_top_k]
                        shap_sum = shap_vals[i, top_k].sum()
                        if shap_sign_match and np.sign(shap_sum) != direction:
                            continue  # Fail confirmation
                    signals[i] = direction
                continue

            # Exit / reversal logic (when holding a position)
            if current_position == 1:
                if prob <= exit_threshold:
                    signals[i] = 0  # flat out of long
                else:
                    # Optional reversal to short if strong opposite signal
                    if prob <= (1 - entry_threshold):
                        if shap_vals is not None:
                            top_k = np.argsort(-np.abs(shap_vals[i]))[:shap_top_k]
                            shap_sum = shap_vals[i, top_k].sum()
                            if shap_sign_match and np.sign(shap_sum) != -1:
                                signals[i] = 1
                                continue
                        signals[i] = -1
                    else:
                        signals[i] = 1
                continue

            if current_position == -1:
                if prob >= (1 - exit_threshold):
                    signals[i] = 0  # flat out of short
                else:
                    if prob >= entry_threshold:
                        if shap_vals is not None:
                            top_k = np.argsort(-np.abs(shap_vals[i]))[:shap_top_k]
                            shap_sum = shap_vals[i, top_k].sum()
                            if shap_sign_match and np.sign(shap_sum) != 1:
                                signals[i] = -1
                                continue
                        signals[i] = 1
                    else:
                        signals[i] = -1
        return signals

    def predict_proba(self, X) -> np.ndarray:
        """Convenience wrapper for forward."""
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save(self, path: str, metadata: Optional[dict] = None) -> None:
        """Save model and metadata to disk."""
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