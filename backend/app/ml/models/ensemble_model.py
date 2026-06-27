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

    The model holds a collection of sub‑models (e.g., LSTM, XGBoost, Lorentzian KNN)
    and optionally a graph‑neural‑network (GNN) signal.  During inference each
    sub‑model produces a probability estimate; these are combined using per‑model
    weights that are normalised on‑the‑fly.  A confidence score derived from the
    aggregated probability is then used to emit a directional signal or a neutral
    label.

    Attributes
    ----------
    model_type : str
        Identifier for the model type; fixed to ``"ensemble"``.
    weights : Dict[str, float]
        Base weights for each sub‑model.  Keys correspond to model names added via
        :meth:`add_model`.  The sum does not need to be 1.0 because normalisation is
        performed at inference time.
    confidence_threshold : float
        Minimum confidence required for a prediction to be considered a directional
        signal (up/down).  Predictions below this threshold are labelled ``"neutral"``.
    gnn_weight : float
        Weight applied to an optional GNN model.  If set to ``0.0`` the GNN is ignored.
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
            Mapping of model name to weight.  If ``None`` a default weighting is used.
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
        single tensor shared across all sub‑models.  If a GNN model is registered and
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
        confidence score.  Predictions whose confidence falls below
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

        # Ensure we iterate over a one‑dimensional batch.  If the model returns a
        # scalar (e.g., a single‑sample prediction), treat it as a batch of size 1.
        if proba.ndim == 0:
            prob_iter = [float(proba)]
            conf_iter = [float(confidence)]
        else:
            prob_iter = proba.tolist()
            conf_iter = confidence.tolist()

        for prob, conf in zip(prob_iter, conf_iter):
            prob_float = float(prob)
            conf_float = float(conf)
            if conf_float < self.confidence_threshold:
                direction = "neutral"
            else:
                direction = "up" if prob_float > 0.5 else "down"
            results.append(
                {
                    "signal": direction,
                    "probability": prob_float,
                    "confidence": conf_float,
                }
            )
        return results

    # -------------------------------------------------------------------------
    # The following helper methods are inherited from ``AbstractModel`` but are
    # re‑exposed here to keep static type checkers happy.  They delegate to the
    # base implementation without alteration.
    # -------------------------------------------------------------------------

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> EvalMetrics:
        """
        Evaluate predictions against true labels using common classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground‑truth binary labels (0 or 1).
        y_pred : np.ndarray
            Predicted probabilities produced by :meth:`forward`.

        Returns
        -------
        EvalMetrics
            Namedtuple containing ``accuracy`` and ``roc_auc`` scores.
        """
        return super().evaluate(y_true, y_pred)

    def save(self, path: Path) -> None:
        """
        Persist the ensemble configuration (weights, threshold, GNN weight) to a JSON
        file.  Sub‑models are responsible for their own persistence.

        Parameters
        ----------
        path : pathlib.Path
            Destination file path.
        """
        data = {
            "weights": self.weights,
            "confidence_threshold": self.confidence_threshold,
            "gnn_weight": self.gnn_weight,
        }
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        """
        Load ensemble configuration from a JSON file written by :meth:`save`.

        Parameters
        ----------
        path : pathlib.Path
            Source file path.
        """
        raw = json.loads(path.read_text())
        self.weights = raw.get("weights", self.weights)
        self.confidence_threshold = raw.get("confidence_threshold", self.confidence_threshold)
        self.gnn_weight = raw.get("gnn_weight", self.gnn_weight)