"""
Weighted ensemble model combining LSTM, XGBoost, Lorentzian KNN, and optional GNN signals.

The ensemble weights are typically optimized on a validation set (e.g., via Optuna) and
only predictions with confidence above a configurable threshold are emitted as actionable
signals.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

logger = structlog.get_logger()


class EnsembleModel(AbstractModel):
    """
    Ensemble model that aggregates predictions from multiple sub‑models.

    Attributes
    ----------
    model_type : str
        Identifier for the model type; fixed to ``"ensemble"``.
    weights : Dict[str, float]
        Base weights for each sub‑model. Keys correspond to model names added via
        :meth:`add_model`. The sum does not need to be 1.0 because normalization is
        performed at inference time.
    confidence_threshold : float
        Minimum confidence required for a prediction to be considered a directional
        signal (up/down). Predictions below this threshold are labeled ``"neutral"``.
    gnn_weight : float
        Weight applied to an optional GNN model. If set to ``0.0`` the GNN is ignored.
    models : Dict[str, AbstractModel]
        Container of registered sub‑models.
    _gnn_model : Optional[Any]
        Optional GNN model instance providing a ``predict`` method.
    """

    model_type = "ensemble"

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.65,
        gnn_weight: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        weights : dict | None
            Mapping of model name to weight. If ``None`` a default weighting is used.
        confidence_threshold : float
            Threshold for confidence filtering.
        gnn_weight : float
            Weight for the optional GNN component.
        """
        self.weights = weights or {
            "lstm": 0.4,
            "xgboost": 0.3,
            "lorentzian": 0.15,
            "ssm": 0.15,
        }
        self.confidence_threshold = confidence_threshold
        self.gnn_weight = gnn_weight
        self.models: Dict[str, AbstractModel] = {}
        self._gnn_model: Optional[Any] = None  # optional GNNSignal instance

    def add_model(self, name: str, model: AbstractModel) -> None:
        """
        Register a sub‑model with the ensemble.

        Parameters
        ----------
        name : str
            Identifier used to retrieve the model's predictions.
        model : AbstractModel
            Instance implementing ``predict``/``predict_proba`` or ``forward``.
        """
        self.models[name] = model

    def register_gnn(self, gnn_model: Any) -> None:
        """
        Register a GNNSignal model to be included in the weighted ensemble.

        When registered, ``gnn_weight`` controls how much the GNN output contributes.
        If ``gnn_weight`` is ``0.0`` (default), the GNN is registered but has no effect
        until the weight is set > 0.

        Parameters
        ----------
        gnn_model : Any
            GNNSignal instance from ``app.ml.models.gnn_signal`` exposing a
            ``predict`` method.
        """
        self._gnn_model = gnn_model

    def forward(self, x: Any) -> np.ndarray:
        """
        Produce an aggregated probability prediction.

        The input ``x`` can be either a dictionary mapping model names to tensors or a
        single tensor shared across all sub‑models. If a GNN model is registered and
        ``gnn_weight`` > 0, its output is blended into the weighted average using the
        configured GNN weight.

        Parameters
        ----------
        x : Any
            Either ``dict[str, Any]`` where keys correspond to model names, or a single
            array/tensor that will be passed to every model.

        Returns
        -------
        np.ndarray
            Ensemble probability predictions with the same shape as the individual
            model outputs.
        """
        predictions: Dict[str, np.ndarray] = {}
        # Per‑call weight overrides — never mutate ``self.weights`` inside forward(),
        # otherwise the GNN weight accumulates across calls and skews normalization.
        call_weights: Dict[str, float] = {}

        for name, model in self.models.items():
            model_input = x[name] if isinstance(x, dict) else x
            try:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(model_input)  # type: ignore[arg-type]
                else:
                    import torch

                    pred = model.forward(
                        model_input
                        if isinstance(model_input, torch.Tensor)
                        else torch.tensor(model_input, dtype=torch.float32)
                    ).numpy()
                predictions[name] = pred
            except Exception:
                # Individual model failures are tolerated; they simply do not contribute.
                continue

        # Include GNN prediction if registered with non‑zero weight
        if self._gnn_model is not None and self.gnn_weight > 0.0:
            try:
                gnn_input = x.get("gnn") if isinstance(x, dict) else None
                if gnn_input is not None:
                    returns_df, node_features = gnn_input
                    gnn_pred = self._gnn_model.predict(returns_df, node_features)
                    predictions["_gnn"] = gnn_pred
                    call_weights["_gnn"] = self.gnn_weight
            except Exception as exc:
                logger.debug("GNN prediction failed in ensemble", error=str(exc))

        if not predictions:
            # Preserve batch shape on the fallback so downstream confidence/threshold
            # logic does not crash on a shape mismatch.
            first = next((p for p in predictions.values()), None)
            return np.full(first.shape if first is not None else 1, 0.5)

        def _w(n: str) -> float:
            return call_weights.get(n, self.weights.get(n, 1.0))

        total_weight = sum(_w(n) for n in predictions) or 1.0
        ensemble = np.zeros(list(predictions.values())[0].shape)
        for name, pred in predictions.items():
            ensemble += (_w(name) / total_weight) * pred

        return ensemble

    def predict_with_confidence(self, x: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both the raw probability and a derived confidence score.

        Confidence is defined as ``2 * |p - 0.5|``, yielding a value in ``[0, 1]`` where
        ``0`` indicates maximum uncertainty and ``1`` indicates full confidence.

        Parameters
        ----------
        x : Any
            Input data passed to :meth:`forward`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(probability, confidence)`` arrays.
        """
        proba = self.forward(x)
        confidence = np.abs(proba - 0.5) * 2  # [0,1] scale: 0=uncertain, 1=very confident
        return proba, confidence

    def predict_signal(self, x: Any) -> List[Dict[str, Any]]:
        """
        Generate a high‑level list of signal dictionaries.

        Each entry contains the predicted direction, the raw probability, and the
        confidence score. Predictions whose confidence falls below
        ``self.confidence_threshold`` are marked as ``"neutral"``.

        Parameters
        ----------
        x : Any
            Input data passed to :meth:`predict_with_confidence`.

        Returns
        -------
        List[dict]
            List of signal dictionaries.
        """
        proba, confidence = self.predict_with_confidence(x)
        results: List[Dict[str, Any]] = []
        for i in range(len(proba)):
            if confidence[i] >= self.confidence_threshold:
                results.append(
                    {
                        "prediction": "up" if proba[i] > 0.5 else "down",
                        "probability": float(proba[i]),
                        "confidence": float(confidence[i]),
                    }
                )
            else:
                results.append(
                    {
                        "prediction": "neutral",
                        "probability": float(proba[i]),
                        "confidence": float(confidence[i]),
                    }
                )
        return results

    def train_epoch(self, loader: Any, optimizer: Any = None, criterion: Any = None) -> Dict[str, float]:
        """
        Placeholder training loop for compatibility with the abstract interface.

        The ensemble itself does not have trainable parameters; concrete sub‑models are
        trained independently. This method returns a dummy metric dictionary.

        Parameters
        ----------
        loader : Any
            Data loader yielding training batches (unused).
        optimizer : Any, optional
            Optimizer instance (unused).
        criterion : Any, optional
            Loss function (unused).

        Returns
        -------
        dict
            Dummy metrics ``{'loss': 0.0, 'accuracy': 0.0}``.
        """
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader: Any) -> EvalMetrics:
        """
        Evaluate the ensemble on a validation set.

        Computes accuracy, ROC‑AUC and returns a placeholder Sharpe ratio.

        Parameters
        ----------
        loader : Any
            Iterable yielding ``(X, y)`` pairs where ``X`` is input data and ``y`` are
            binary labels.

        Returns
        -------
        EvalMetrics
            Dataclass containing ``accuracy``, ``auc`` and ``sharpe`` values.
        """
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        for X, y in loader:
            probs = self.forward(X)
            all_probs.append(probs)
            all_labels.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
        probs_cat = np.concatenate(all_probs)
        labels_cat = np.concatenate(all_labels)
        preds = (probs_cat > 0.5).astype(int)
        acc = float(accuracy_score(labels_cat, preds))
        try:
            auc = float(roc_auc_score(labels_cat, probs_cat))
        except ValueError:
            auc = 0.5
        return EvalMetrics(accuracy=acc, auc=auc, sharpe=0.0)

    def optimize_weights_walk_forward(
        self,
        returns_by_model: Dict[str, "pd.Series"],
        actual_returns: "pd.Series",
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        Walk‑forward ensemble weight optimization.

        Uses SciPy's SLSQP optimizer to find weights that maximise Sharpe ratio on each
        fold, then returns the average weights across folds.

        Parameters
        ----------
        returns_by_model : dict[str, pd.Series]
            Mapping of model name to predicted‑return series (identical index).
        actual_returns : pd.Series
            Actual forward returns series aligned with the predictions.
        n_splits : int, default 5
            Number of walk‑forward folds.

        Returns
        -------
        dict[str, float]
            Optimized weights for each model, normalised to sum to 1.
        """
        import pandas as pd
        from scipy.optimize import minimize

        model_names = list(returns_by_model.keys())
        if len(model_names) < 2:
            # Degenerate case – distribute weight equally (or 1.0 if only one model)
            return {k: 1.0 / max(len(model_names), 1) for k in model_names}

        # Align all series to a common index and drop any NaNs
        pred_df = pd.DataFrame(returns_by_model).dropna()
        actual = actual_returns.reindex(pred_df.index).dropna()
        pred_df = pred_df.loc[actual.index]

        n = len(pred_df)
        if n < n_splits * 10:
            # Not enough data for a stable walk‑forward; fall back to equal weighting
            return {k: 1.0 / len(model_names) for k in model_names}

        fold_size = n // n_splits
        all_weights: List[Dict[str, float]] = []

        def neg_sharpe(w: np.ndarray, preds: np.ndarray, actual_arr: np.ndarray) -> float:
            """
            Negative Sharpe objective for the optimizer.

            Parameters
            ----------
            w : np.ndarray
                Weight vector for the models.
            preds : np.ndarray
                Matrix of predicted returns (samples × models).
            actual_arr : np.ndarray
                Vector of actual returns.

            Returns
            -------
            float
                Negative Sharpe ratio (to be minimised).
            """
            portfolio_ret = preds @ w
            excess = portfolio_ret - actual_arr
            std = excess.std()
            if std < 1e-8:
                return 0.0
            return -(excess.mean() / std * np.sqrt(252))

        for fold in range(n_splits):
            train_end = (fold + 1) * fold_size
            if train_end > n:
                break
            preds = pred_df.values[:train_end]
            act = actual.values[:train_end]

            n_models = len(model_names)
            w0 = np.ones(n_models) / n_models
            bounds = [(0.0, 1.0)] * n_models
            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

            try:
                result = minimize(
                    neg_sharpe,
                    w0,
                    args=(preds, act),
                    bounds=bounds,
                    constraints=constraints,
                    method="SLSQP",
                )
                if result.success:
                    weight_dict = dict(zip(model_names, result.x))
                else:
                    weight_dict = dict(zip(model_names, w0))
            except Exception as exc:
                logger.debug("Weight optimization failed for fold", error=str(exc))
                weight_dict = dict(zip(model_names, w0))

            all_weights.append(weight_dict)

        # Average the weights across successful folds
        avg_weights: Dict[str, float] = {k: 0.0 for k in model_names}
        for w in all_weights:
            for k, v in w.items():
                avg_weights[k] += v
        num_folds = len(all_weights) or 1
        avg_weights = {k: v / num_folds for k, v in avg_weights.items()}

        # Normalise to ensure sum equals 1.0 (guard against numerical drift)
        total = sum(avg_weights.values())
        if total > 0:
            avg_weights = {k: v / total for k, v in avg_weights.items()}

        return avg_weights