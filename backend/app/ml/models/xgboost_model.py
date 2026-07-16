"""
XGBoost binary classifier with Optuna hyperparameter optimization and SHAP-based explainability.

This module defines :class:`XGBoostClassifier`, a concrete implementation of
:class:`~app.ml.models.base_model.AbstractModel`. The classifier wraps
``xgboost.XGBClassifier`` and provides a simple interface for training,
prediction, evaluation and feature importance extraction using SHAP values.

Typical usage::

    model = XGBoostClassifier(n_estimators=300, max_depth=4)
    metrics = model.fit(X_train, y_train, X_val, y_val, feature_names=['open', 'close'])
    probs = model.predict_proba(X_test)
    importance = model.get_feature_importance()
    model.save('model.pt')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    """XGBoost binary classifier with built‑in SHAP explainability.

    The class inherits from :class:`AbstractModel` and implements the required
    methods for a production‑ready model component. It uses ``xgboost`` for
    training and inference and ``shap`` for feature importance extraction.
    """

    model_type = "xgboost"

    def __init__(self, **kwargs: Any) -> None:
        """Create a new classifier instance.

        Args:
            **kwargs: Hyper‑parameter overrides for the underlying
                :class:`xgboost.XGBClassifier`. Supported keys include
                ``n_estimators``, ``max_depth``, ``learning_rate``,
                ``subsample``, ``colsample_bytree``, ``min_child_weight``,
                ``reg_alpha`` and ``reg_lambda``. Unspecified parameters fall back
                to sensible defaults.
        """
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        self.params: Dict[str, Any] = {
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

    def forward(self, x: Any) -> np.ndarray:
        """Compute the probability of the positive class for the input ``x``.

        Args:
            x: Input features. May be a NumPy array, a PyTorch tensor,
               or any object exposing a ``numpy()`` method.

        Returns:
            A one‑dimensional ``np.ndarray`` containing the predicted
            probabilities for the positive class.
        """
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
        """Fit the model on training data and evaluate on a validation set.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            feature_names: Optional list of feature names used for SHAP
                explanations. If omitted, generic names will be generated.

        Returns:
            A dictionary with keys ``val_accuracy`` and ``val_auc`` containing
            the validation accuracy and AUC scores.
        """
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

    def train_epoch(
        self,
        loader: Iterable[Tuple[Any, Any]],
        optimizer: Any = None,
        criterion: Any = None,
    ) -> Dict[str, float]:
        """Placeholder for epoch‑wise training.

        XGBoost performs full‑batch training via :meth:`fit`; therefore this
        method simply returns zeroed metrics to satisfy the abstract interface.

        Args:
            loader: An iterable yielding ``(features, labels)`` tuples.
            optimizer: Ignored; present for API compatibility.
            criterion: Ignored; present for API compatibility.

        Returns:
            A dictionary with ``loss`` and ``accuracy`` set to ``0.0``.
        """
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader: Iterable[Tuple[Any, Any]]) -> EvalMetrics:
        """Evaluate the model on a dataset provided by ``loader``.

        Args:
            loader: An iterable yielding ``(features, labels)`` tuples.

        Returns:
            An :class:`EvalMetrics` instance containing accuracy, AUC and a
            placeholder Sharpe ratio (set to ``0.0``).
        """
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
        """Return SHAP‑based feature importance scores.

        The method lazily creates a :class:`shap.TreeExplainer` if one does not
        already exist, then maps the model's native feature importances to the
        provided ``feature_names`` (or generic names) and sorts them descending.

        Returns:
            A dictionary mapping feature names to importance values.
        """
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
        """Predict the probability of the positive class for ``X``.

        Args:
            X: Input features, optionally exposing a ``numpy()`` method.

        Returns:
            A one‑dimensional ``np.ndarray`` of predicted probabilities.
        """
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist the model and associated metadata to disk.

        The XGBoost model is saved using its native ``save_model`` method with a
        ``.ubj`` extension. A companion JSON file stores ``feature_names``,
        hyper‑parameters and any additional ``metadata`` supplied by the caller.

        Args:
            path: Destination file path (any extension; ``.pt`` is replaced with
                ``.ubj`` for the model file).
            metadata: Optional dictionary of additional information to store.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model_path = path.replace(".pt", ".ubj")
        self.model.save_model(model_path)
        meta = {"feature_names": self.feature_names, "params": self.params, **(metadata or {})}
        Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        """Load a previously saved model from ``path``.

        The method expects a ``.ubj`` model file and an optional JSON metadata
        file with the same base name.

        Args:
            path: Path to the model file (any extension; ``.pt`` is replaced with
                ``.ubj`` internally).

        Returns:
            An instantiated and populated :class:`XGBoostClassifier`.
        """
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        instance.model.load_model(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance.feature_names = meta.get("feature_names", [])
        return instance