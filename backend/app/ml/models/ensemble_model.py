"""
Weighted ensemble of LSTM + XGBoost + Lorentzian KNN.
Weights optimized on validation set via Optuna.
Only signals with confidence > threshold are forwarded.
"""

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
    weights: dict[str, float]
        Mapping of model identifiers to their respective contribution weights.
        The weights should be non‑negative and sum to 1.0 (within a tolerance).
        Example: ``{"lstm": 0.5, "xgboost": 0.35, "lorentzian": 0.15}``.
    confidence_threshold: float
        Minimum confidence required for a prediction to be emitted as a directional
        signal. Must be in the interval ``[0.0, 1.0]``. Example: ``0.65``.
    gnn_weight: float
        Optional contribution weight for a registered GNN model. Must be non‑negative.
        If ``0.0`` the GNN has no effect even when registered. Example: ``0.0``.
    """

    weights: dict[str, float] = Field(
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
    def _validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
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
    def _check_consistency(self):
        # All field constraints are already enforced by Pydantic.
        return self


class EnsembleModel(AbstractModel):
    model_type = "ensemble"

    def __init__(
        self,
        weights: dict | None = None,
        confidence_threshold: float = 0.65,
        gnn_weight: float = 0.0,
    ):
        # Validate and normalise configuration via Pydantic schema
        config = EnsembleConfig(
            weights=weights or {"lstm": 0.5, "xgboost": 0.35, "lorentzian": 0.15},
            confidence_threshold=confidence_threshold,
            gnn_weight=gnn_weight,
        )
        self.weights: dict[str, float] = config.weights
        self.confidence_threshold: float = config.confidence_threshold
        self.gnn_weight: float = config.gnn_weight

        self.models: dict[str, AbstractModel] = {}
        self._gnn_model = None  # optional GNNSignal instance

    def add_model(self, name: str, model: AbstractModel) -> None:
        """Register a sub‑model under a given name."""
        self.models[name] = model

    def register_gnn(self, gnn_model) -> None:
        """
        Register a GNNSignal model to be included in the weighted ensemble.

        When registered, ``gnn_weight`` controls how much the GNN output contributes.
        If ``gnn_weight`` is ``0.0`` (default), the GNN is registered but has no effect
        until ``gnn_weight`` is set > 0.

        Args
        ----
        gnn_model
            GNNSignal instance from ``app.ml.models.gnn_signal``.
        """
        self._gnn_model = gnn_model

    def forward(self, x) -> np.ndarray:
        """
        Compute the ensemble prediction.

        Parameters
        ----------
        x : dict | Any
            Either a mapping ``{model_name: tensor}`` providing model‑specific inputs,
            or a single tensor that will be broadcast to all models.

        Returns
        -------
        np.ndarray
            Weighted ensemble probability vector.
        """
        predictions: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            model_input = x[name] if isinstance(x, dict) else x
            try:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(model_input)
                else:
                    import torch

                    pred = model.forward(
                        model_input
                        if isinstance(model_input, torch.Tensor)
                        else torch.tensor(model_input, dtype=torch.float32)
                    ).numpy()
                predictions[name] = pred
            except Exception:
                # Individual model failure should not abort the whole ensemble.
                logger.debug("Model prediction failed", model=name, error=str(predictions.get(name)))
                continue

        # Include GNN prediction if registered with non‑zero weight
        if self._gnn_model is not None and self.gnn_weight > 0.0:
            try:
                if isinstance(x, dict) and "gnn" in x:
                    returns_df, node_features = x["gnn"]
                    gnn_pred = self._gnn_model.predict(returns_df, node_features)
                    predictions["_gnn"] = gnn_pred
                    # Use a copy of weights to avoid mutating the original dict.
                    self.weights = {**self.weights, "_gnn": self.gnn_weight}
            except Exception as exc:
                logger.debug("GNN prediction failed in ensemble", error=str(exc))

        if not predictions:
            # No predictions available – return a neutral probability.
            return np.full(1, 0.5)

        # Normalise weights so they sum to 1 for the present predictions.
        total_weight = sum(self.weights.get(n, 1.0) for n in predictions)
        ensemble = np.zeros_like(next(iter(predictions.values())))
        for name, pred in predictions.items():
            w = self.weights.get(name, 1.0) / total_weight
            ensemble += w * pred

        return ensemble

    def predict_with_confidence(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Return (direction probability, confidence) arrays."""
        proba = self.forward(x)
        confidence = np.abs(proba - 0.5) * 2  # [0,1] scale: 0=uncertain, 1=very confident
        return proba, confidence

    def predict_signal(self, x) -> list[dict]:
        """High‑level: return list of signal dicts above confidence threshold."""
        proba, confidence = self.predict_with_confidence(x)
        results: list[dict] = []
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

    def train_epoch(self, loader, optimizer=None, criterion=None) -> dict:
        """Placeholder training step – returns dummy metrics."""
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
        """Evaluate the ensemble on a data loader and return aggregated metrics."""
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for X, y in loader:
            probs = self.forward(X)
            all_probs.append(probs)

            if hasattr(y, "numpy"):
                labels = y.numpy()
            else:
                labels = np.asarray(y)
            all_labels.append(labels)

        if not all_probs:
            raise ValueError("Loader produced no data for evaluation.")

        probs_concat = np.concatenate(all_probs)
        labels_concat = np.concatenate(all_labels)

        # Binary classification metrics
        pred_labels = (probs_concat > 0.5).astype(int)
        accuracy = accuracy_score(labels_concat, pred_labels)
        try:
            roc_auc = roc_auc_score(labels_concat, probs_concat)
        except ValueError:
            # ROC AUC is undefined if only one class is present.
            roc_auc = float("nan")

        return EvalMetrics(accuracy=accuracy, roc_auc=roc_auc)