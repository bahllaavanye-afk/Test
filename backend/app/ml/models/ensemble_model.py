"""
Weighted ensemble of LSTM + XGBoost + Lorentzian KNN.
Weights optimized on validation set via Optuna.
Only signals with confidence > threshold are forwarded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, roc_auc_score
from pydantic import BaseModel, Field, field_validator, model_validator

from app.ml.models.base_model import AbstractModel, EvalMetrics

logger = structlog.get_logger()


class EnsembleConfig(BaseModel):
    """Configuration schema for :class:`EnsembleModel`.

    Attributes
    ----------
    weights:
        Mapping of model identifiers to their respective contribution weights.
        The weights should be non‑negative and sum to 1.0 (within a tolerance).
        Example: ``{"lstm": 0.5, "xgboost": 0.35, "lorentzian": 0.15}``.
    confidence_threshold:
        Minimum confidence required for a prediction to be emitted as a directional
        signal. Must be in the interval ``[0.0, 1.0]``.
    gnn_weight:
        Optional contribution weight for a registered GNN model. Must be non‑negative.
        If ``0.0`` the GNN has no effect even when registered.
    """

    weights: Dict[str, float] = Field(
        default_factory=lambda: {"lstm": 0.5, "xgboost": 0.35, "lorentzian": 0.15},
        description="Model name to weight mapping; values should be non‑negative and sum to 1.0.",
        json_schema_extra={"example": {"lstm": 0.5, "xgboost": 0.35, "lorentzian": 0.15}},
    )
    confidence_threshold: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for emitting a directional signal.",
        json_schema_extra={"example": 0.65},
    )
    gnn_weight: float = Field(
        0.0,
        ge=0.0,
        description="Weight given to the optional GNN model in the ensemble.",
        json_schema_extra={"example": 0.0},
    )

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that weight values are non‑negative and sum to 1.0."""
        if not v:
            raise ValueError("weights dictionary must contain at least one entry")
        for name, w in v.items():
            if w < 0.0:
                raise ValueError(f"weight for '{name}' must be non‑negative, got {w}")
        total = sum(v.values())
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(f"weights must sum to 1.0 (got {total:.6f})")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "EnsembleConfig":
        """Placeholder for cross‑field validation; currently a no‑op."""
        return self


class EnsembleModel(AbstractModel):
    """A weighted ensemble that combines predictions from multiple sub‑models.

    The ensemble supports an optional GNN model that can be registered after
    instantiation.  Predictions are combined using the configured weights and
    a confidence score is derived from the distance to the 0.5 decision boundary.
    """

    model_type = "ensemble"

    def __init__(
        self,
        weights: Union[Dict[str, float], None] = None,
        confidence_threshold: float = 0.65,
        gnn_weight: float = 0.0,
    ) -> None:
        """Create a new :class:`EnsembleModel` instance.

        Parameters
        ----------
        weights:
            Optional custom weight mapping.  If omitted the default mapping is used.
        confidence_threshold:
            Threshold for the confidence score; predictions below this are labelled
            ``neutral``.
        gnn_weight:
            Weight for a registered GNN model.  A value of ``0.0`` disables the GNN
            contribution even if a model is registered.
        """
        config = EnsembleConfig(
            weights=weights or {"lstm": 0.5, "xgboost": 0.35, "lorentzian": 0.15},
            confidence_threshold=confidence_threshold,
            gnn_weight=gnn_weight,
        )
        self.weights: Dict[str, float] = config.weights
        self.confidence_threshold: float = config.confidence_threshold
        self.gnn_weight: float = config.gnn_weight

        self.models: Dict[str, AbstractModel] = {}
        self._gnn_model: Any = None  # optional GNNSignal instance

    def add_model(self, name: str, model: AbstractModel) -> None:
        """Register a sub‑model under a given name."""
        self.models[name] = model

    def register_gnn(self, gnn_model: Any) -> None:
        """Register a GNNSignal model to be included in the weighted ensemble.

        When registered, ``gnn_weight`` controls how much the GNN output contributes.
        If ``gnn_weight`` is ``0.0`` (default), the GNN is registered but has no effect
        until ``gnn_weight`` is set > 0.

        Parameters
        ----------
        gnn_model:
            GNNSignal instance from ``app.ml.models.gnn_signal``.
        """
        self._gnn_model = gnn_model

    def forward(self, x: Union[Dict[str, Any], Any]) -> np.ndarray:
        """Compute the ensemble prediction.

        Parameters
        ----------
        x:
            Either a mapping ``{model_name: tensor}`` providing model‑specific inputs,
            or a single tensor that will be broadcast to all models.

        Returns
        -------
        np.ndarray
            Weighted ensemble probability vector.
        """
        predictions: Dict[str, np.ndarray] = {}
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
                continue

        # Include GNN prediction if registered with non‑zero weight
        if self._gnn_model is not None and self.gnn_weight > 0.0:
            try:
                gnn_input = x.get("gnn") if isinstance(x, dict) else None
                if gnn_input is not None:
                    returns_df, node_features = gnn_input
                    gnn_pred = self._gnn_model.predict(returns_df, node_features)
                    predictions["_gnn"] = gnn_pred
                    self.weights["_gnn"] = self.gnn_weight
            except Exception as exc:
                logger.debug("GNN prediction failed in ensemble", error=str(exc))

        if not predictions:
            return np.full(1, 0.5)

        total_weight = sum(self.weights.get(n, 1.0) for n in predictions)
        ensemble = np.zeros(list(predictions.values())[0].shape, dtype=float)
        for name, pred in predictions.items():
            w = self.weights.get(name, 1.0) / total_weight
            ensemble += w * pred

        return ensemble

    def predict_with_confidence(self, x: Union[Dict[str, Any], Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Return probability and confidence arrays for the given input.

        Confidence is defined as ``2 * |p - 0.5|`` and lies in ``[0, 1]``.

        Parameters
        ----------
        x:
            Input data passed to :meth:`forward`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(probability, confidence)`` arrays.
        """
        proba = self.forward(x)
        confidence = np.abs(proba - 0.5) * 2
        return proba, confidence

    def predict_signal(self, x: Union[Dict[str, Any], Any]) -> List[Dict[str, Any]]:
        """Return a list of signal dictionaries, applying the confidence threshold.

        Each element corresponds to a time step (or sample) in the input and
        contains the predicted direction, the raw probability, and the confidence
        score.  If the confidence is below ``self.confidence_threshold`` the
        direction is reported as ``neutral``.

        Parameters
        ----------
        x:
            Input data passed to :meth:`forward`.

        Returns
        -------
        List[Dict[str, Any]]
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

    def train_epoch(self, loader: Iterable[Any], optimizer: Any = None, criterion: Any = None) -> Dict[str, float]:
        """Placeholder training step – returns dummy metrics.

        Parameters
        ----------
        loader:
            Iterable yielding training batches.
        optimizer:
            Optimiser instance (unused in the placeholder implementation).
        criterion:
            Loss function (unused in the placeholder implementation).

        Returns
        -------
        dict
            Dummy metric dictionary containing ``loss`` and ``accuracy``.
        """
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader: Iterable[Tuple[Any, Any]]) -> EvalMetrics:
        """Evaluate the ensemble on a data loader and return aggregated metrics.

        The method aggregates probabilities and true labels across all batches,
        computes classification accuracy and ROC‑AUC (when applicable), and
        returns the results wrapped in an :class:`EvalMetrics` instance.

        Parameters
        ----------
        loader:
            Iterable yielding ``(X, y)`` pairs where ``X`` is the input data and
            ``y`` is a tensor/array of binary labels.

        Returns
        -------
        EvalMetrics
            Structured evaluation metrics.
        """
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for X, y in loader:
            probs = self.forward(X)
            all_probs.append(probs.squeeze())
            # ``y`` may be a torch tensor or a numpy array
            if hasattr(y, "numpy"):
                all_labels.append(y.numpy().squeeze())
            else:
                all_labels.append(np.asarray(y).squeeze())

        if not all_probs:
            raise ValueError("Evaluation loader yielded no data")

        probs_concat = np.concatenate(all_probs)
        labels_concat = np.concatenate(all_labels)

        # Binary predictions from probability > 0.5
        preds = (probs_concat > 0.5).astype(int)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(labels_concat, preds)),
        }

        # ROC‑AUC is only defined for binary classification with both classes present
        try:
            if len(np.unique(labels_concat)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(labels_concat, probs_concat))
        except Exception as exc:  # pragma: no cover
            logger.debug("ROC‑AUC calculation failed", error=str(exc))

        return EvalMetrics(**metrics)