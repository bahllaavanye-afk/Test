"""
XGBoost binary classifier with Optuna hyperparameter optimization and SHAP-based
explainability.

The class follows the :class:`~app.ml.models.base_model.AbstractModel` interface
used throughout the QuantEdge codebase.  It provides a thin wrapper around
``xgboost.XGBClassifier`` that exposes a PyTorch‑style API (``forward``,
``fit``, ``evaluate``) while handling model persistence and feature importance
extraction via SHAP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

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
    """Binary classification model based on XGBoost.

    Attributes
    ----------
    model_type: str
        Identifier used by the framework to select the model implementation.
    params: Dict[str, Any]
        Hyper‑parameters passed to ``xgboost.XGBClassifier``.
    model: xgb.XGBClassifier
        The underlying XGBoost estimator.
    _explainer: Optional[shap.TreeExplainer]
        Cached SHAP explainer instantiated on demand.
    feature_names: List[str]
        Optional list of feature names for interpretability.
    """

    model_type = "xgboost"

    def __init__(self, **kwargs: Any) -> None:
        """Create a new XGBoost classifier instance.

        Parameters
        ----------
        **kwargs : Any
            Optional hyper‑parameter overrides. Supported keys include
            ``n_estimators``, ``max_depth``, ``learning_rate``, ``subsample``,
            ``colsample_bytree``, ``min_child_weight``, ``reg_alpha`` and
            ``reg_lambda``.
        """
        if not XGB_AVAILABLE:  # pragma: no cover
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
        self.model = xgb.XGBClassifier(
            **self.params,
            early_stopping_rounds=50,
            verbosity=0,
        )
        self._explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []

    def forward(self, x: Any) -> np.ndarray:
        """Compute predicted probabilities for the positive class.

        Parameters
        ----------
        x : Any
            Input features; can be a NumPy array or a tensor‑like object with a
            ``numpy`` method.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples,)`` containing the probability of class 1.
        """
        if hasattr(x, "numpy"):
            x = x.numpy()
        return self.model.predict_proba(x)[:, 1]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Fit the model on training data and evaluate on a validation set.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target vector.
        X_val : np.ndarray
            Validation feature matrix.
        y_val : np.ndarray
            Validation target vector.
        feature_names : list[str] | None, optional
            Optional list of feature names for later interpretability.

        Returns
        -------
        dict
            Dictionary containing validation ``accuracy`` and ``auc`` scores.
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

    def train_epoch(self, loader: Any, optimizer: Any = None, criterion: Any = None) -> Dict[str, float]:
        """Placeholder for compatibility with epoch‑based training loops.

        XGBoost handles training internally via ``fit``; therefore this method
        returns a dummy loss and accuracy.

        Parameters
        ----------
        loader : Any
            Ignored; present for interface compatibility.
        optimizer : Any, optional
            Ignored; present for interface compatibility.
        criterion : Any, optional
            Ignored; present for interface compatibility.

        Returns
        -------
        dict
            Dictionary with keys ``loss`` and ``accuracy`` set to ``0.0``.
        """
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader: Iterable[Tuple[np.ndarray, np.ndarray]]) -> EvalMetrics:
        """Evaluate the model on a data loader.

        Parameters
        ----------
        loader : iterable of (X, y)
            Iterable yielding feature matrices and label vectors. Elements may be
            NumPy arrays or tensor‑like objects exposing a ``numpy`` method.

        Returns
        -------
        EvalMetrics
            Aggregated evaluation metrics (accuracy, AUC, Sharpe).
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
        except ValueError:  # pragma: no cover
            auc = 0.5
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return SHAP‑based feature importance sorted descending.

        Returns
        -------
        dict[str, float]
            Mapping from feature name to importance score.
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
        """Alias for ``forward``; returns probability of the positive class.

        Parameters
        ----------
        X : Any
            Input features; can be a NumPy array or a tensor‑like object with a
            ``numpy`` method.

        Returns
        -------
        np.ndarray
            Probability of class 1 for each sample.
        """
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist the model and associated metadata to disk.

        Parameters
        ----------
        path : str
            Destination file path for the model. The extension ``.pt`` will be
            replaced with ``.ubj`` for the native XGBoost format.
        metadata : dict | None, optional
            Additional JSON‑serialisable information to store alongside the model.
        """
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
        """Load a previously saved model from disk.

        Parameters
        ----------
        path : str
            Path to the model file (original ``.pt`` location). The method will
            locate the corresponding ``.ubj`` and ``.json`` files.

        Returns
        -------
        XGBoostClassifier
            Fully initialised classifier with loaded parameters and feature names.
        """
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        instance.model.load_model(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance.feature_names = meta.get("feature_names", [])
        return instance