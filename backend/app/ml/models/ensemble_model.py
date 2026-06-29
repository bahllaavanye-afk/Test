"""
Weighted ensemble of LSTM + XGBoost + Lorentzian KNN.
Weights optimized on validation set via Optuna.
Only signals with confidence > threshold are forwarded.
"""

import json
import numpy as np
import structlog
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from pydantic import BaseModel, Field, validator, root_validator

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
        example={"lstm": 0.5, "xgboost": 0.35, "lorentzian": 0.15},
    )
    confidence_threshold: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for emitting a directional signal.",
        example=0.65,
    )
    gnn_weight: float = Field(
        0.0,
        ge=0.0,
        description="Weight given to the optional GNN model in the ensemble.",
        example=0.0,
    )

    @validator("weights")
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

    @root_validator(skip_on_failure=True)
    def _check_consistency(cls, values):
        # confidence_threshold already bounded by Field; gnn_weight by Field.
        # Additional cross‑field checks could be added here if needed.
        return values


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
        predictions = {}
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
        ensemble = np.zeros(list(predictions.values())[0].shape)
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
        results = []
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
        all_probs, all_labels = [], []
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
        returns_by_model: dict[str, "pd.Series"],
        actual_returns: "pd.Series",
        n_splits: int = 5,
    ) -> dict[str, float]:
        """
        Walk‑forward ensemble weight optimization.

        Uses SciPy SLSQP to find weights that maximise Sharpe on each fold,
        then returns the average weights across folds.

        Parameters
        ----------
        returns_by_model : dict[str, pd.Series]
            Mapping of model name → predicted‑return series (identical index).
        actual_returns : pd.Series
            Actual forward returns series (identical index as above).
        n_splits : int, optional
            Number of walk‑forward folds (default ``5``).

        Returns
        -------
        dict[str, float]
            Optimised weight per model, summing to 1.
        """
        import pandas as pd
        import numpy as np
        from scipy.optimize import minimize

        model_names = list(returns_by_model.keys())
        if len(model_names) < 2:
            return {k: 1.0 / max(len(model_names), 1) for k in model_names}

        # Align all series to common index
        pred_df = pd.DataFrame(returns_by_model).dropna()
        actual = actual_returns.reindex(pred_df.index).dropna()
        pred_df = pred_df.loc[actual.index]

        n = len(pred_df)
        if n < n_splits * 10:
            return {k: 1.0 / len(model_names) for k in model_names}

        fold_size = n // n_splits
        all_weights = []

        def neg_sharpe(w: np.ndarray, preds: np.ndarray, actual_arr: np.ndarray) -> float:
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
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 200, "ftol": 1e-8},
                )
                if result.success:
                    w = np.maximum(result.x, 0.0)
                    w = w / w.sum()
                    all_weights.append(w)
            except Exception as exc:
                logger.debug("Weight optimization fold failed", error=str(exc))

        if not all_weights:
            # Fallback to equal weighting if optimisation failed
            return {k: 1.0 / len(model_names) for k in model_names}

        avg_weights = np.mean(all_weights, axis=0)
        return dict(zip(model_names, avg_weights.tolist()))