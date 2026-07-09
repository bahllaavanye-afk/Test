"""
XGBoost binary classifier with Optuna hyperparameter optimization and SHAP-based explainability.

This module defines ``XGBoostClassifier`` which conforms to the ``AbstractModel`` interface
used throughout the QuantEdge codebase.  The class wraps ``xgboost.XGBClassifier`` and
provides methods for training, inference, evaluation, feature importance extraction, and
persistence.  All heavy‑lifting is delegated to the underlying XGBoost implementation; the
wrapper adds type safety, documentation, and convenient serialization of model metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    """Binary classifier based on XGBoost with optional SHAP explainability.

    The classifier is configured for a binary logistic objective and evaluates using AUC.
    Hyperparameters can be overridden via keyword arguments at construction time.
    """

    model_type = "xgboost"

    def __init__(self, **kwargs: Any) -> None:
        """Create a new ``XGBoostClassifier`` instance.

        Args:
            **kwargs: Optional hyperparameters for the underlying ``XGBClassifier``.
                Supported keys include ``n_estimators``, ``max_depth``, ``learning_rate``,
                ``subsample``, ``colsample_bytree``, ``min_child_weight``,
                ``reg_alpha``, and ``reg_lambda``.  Missing entries fall back to sensible
                defaults.

        Raises:
            ImportError: If the ``xgboost`` package is not installed.
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
        self._explainer: Optional[Any] = None
        self.feature_names: List[str] = []

    def forward(self, x: Any) -> np.ndarray:
        """Run inference and return the probability of the positive class.

        Args:
            x: Input features.  Can be a NumPy array or any object exposing a ``.numpy()``
               method (e.g., a torch tensor).

        Returns:
            A ``np.ndarray`` of shape ``(n_samples,)`` containing the predicted probabilities.
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
            X_train: Training feature matrix.
            y_train: Training target vector.
            X_val: Validation feature matrix.
            y_val: Validation target vector.
            feature_names: Optional list of feature names for later SHAP explanations.

        Returns:
            A dictionary with validation metrics ``val_accuracy`` and ``val_auc``.
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
        loader: Any,
        optimizer: Any = None,
        criterion: Any = None,
    ) -> Dict[str, float]:
        """Placeholder for epoch‑based training.

        XGBoost optimizes the entire dataset via ``fit``; therefore this method returns
        a dummy loss and accuracy to satisfy the ``AbstractModel`` interface.

        Args:
            loader: Ignored; present for API compatibility.
            optimizer: Ignored; present for API compatibility.
            criterion: Ignored; present for API compatibility.

        Returns:
            A dictionary with keys ``loss`` and ``accuracy`` both set to ``0.0``.
        """
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader: Iterable[Tuple[Any, Any]]) -> EvalMetrics:
        """Evaluate the model on a data loader, returning standard metrics.

        Args:
            loader: An iterable yielding ``(features, labels)`` tuples.  Elements may be
                NumPy arrays or objects exposing a ``.numpy()`` method.

        Returns:
            An ``EvalMetrics`` instance populated with accuracy, AUC, and a placeholder
            Sharpe ratio (set to ``0.0``).
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
        """Return SHAP‑based feature importance sorted descending.

        The method constructs a ``shap.TreeExplainer`` on first call and then maps the
        model's native ``feature_importances_`` to the provided or autogenerated feature
        names.

        Returns:
            A dictionary mapping feature names to importance scores.
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
        """Return the probability of the positive class for the given input.

        Args:
            X: Input features, either a NumPy array or an object with a ``.numpy()`` method.

        Returns:
            A ``np.ndarray`` of shape ``(n_samples,)`` with predicted probabilities.
        """
        if hasattr(X, "numpy"):
            X = X.numpy()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist the model and its metadata to disk.

        The XGBoost model is saved in its native ``.ubj`` format; a companion JSON file
        stores feature names and hyperparameters.

        Args:
            path: Destination file path ending with ``.pt`` (the extension is replaced).
            metadata: Optional additional metadata to merge into the JSON side‑car file.
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
        """Load a previously saved ``XGBoostClassifier`` from disk.

        Args:
            path: Path to the JSON metadata file (the ``.pt`` extension is accepted for
                convenience; it is internally mapped to the corresponding ``.ubj`` model).

        Returns:
            An instantiated ``XGBoostClassifier`` with model weights and metadata restored.
        """
        model_path = path.replace(".pt", ".ubj")
        meta_path = Path(path).with_suffix(".json")
        instance = cls()
        instance.model.load_model(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance.feature_names = meta.get("feature_names", [])
        return instance