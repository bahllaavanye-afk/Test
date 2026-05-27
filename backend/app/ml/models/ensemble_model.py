"""
Weighted ensemble of LSTM + XGBoost + Lorentzian KNN.
Weights optimized on validation set via Optuna.
Only signals with confidence > threshold are forwarded.
"""
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from app.ml.models.base_model import AbstractModel, EvalMetrics


class EnsembleModel(AbstractModel):
    model_type = "ensemble"

    def __init__(
        self,
        weights: dict | None = None,
        confidence_threshold: float = 0.65,
        gnn_weight: float = 0.0,
    ):
        self.weights = weights or {"lstm": 0.5, "xgboost": 0.35, "lorentzian": 0.15}
        self.confidence_threshold = confidence_threshold
        self.gnn_weight = gnn_weight
        self.models: dict[str, AbstractModel] = {}
        self._gnn_model = None  # optional GNNSignal instance

    def add_model(self, name: str, model: AbstractModel) -> None:
        self.models[name] = model

    def register_gnn(self, gnn_model) -> None:
        """
        Register a GNNSignal model to be included in the weighted ensemble.
        When registered, gnn_weight controls how much the GNN output contributes.
        If gnn_weight is 0.0 (default), the GNN is registered but has no effect
        until gnn_weight is set > 0.

        Args:
            gnn_model: GNNSignal instance from app.ml.models.gnn_signal
        """
        self._gnn_model = gnn_model

    def forward(self, x) -> np.ndarray:
        """
        x can be dict of {model_name: tensor} or single tensor shared across all.
        When a GNN model is registered and gnn_weight > 0, its output is blended
        into the weighted average using gnn_weight as its contribution weight.
        """
        predictions = {}
        for name, model in self.models.items():
            model_input = x[name] if isinstance(x, dict) else x
            try:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(model_input)
                else:
                    import torch
                    pred = model.forward(model_input if isinstance(model_input, torch.Tensor)
                                        else torch.tensor(model_input, dtype=torch.float32)).numpy()
                predictions[name] = pred
            except Exception:
                continue

        # Include GNN prediction if registered with non-zero weight
        if self._gnn_model is not None and self.gnn_weight > 0.0:
            try:
                gnn_input = x.get("gnn") if isinstance(x, dict) else None
                if gnn_input is not None:
                    returns_df, node_features = gnn_input
                    gnn_pred = self._gnn_model.predict(returns_df, node_features)
                    predictions["_gnn"] = gnn_pred
                    self.weights["_gnn"] = self.gnn_weight
            except Exception:
                pass

        if not predictions:
            return np.full(1, 0.5)

        total_weight = sum(self.weights.get(n, 1.0) for n in predictions)
        ensemble = np.zeros(list(predictions.values())[0].shape)
        for name, pred in predictions.items():
            w = self.weights.get(name, 1.0) / total_weight
            ensemble += w * pred

        return ensemble

    def predict_with_confidence(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Returns (direction_proba, confidence) arrays."""
        proba = self.forward(x)
        confidence = np.abs(proba - 0.5) * 2   # [0,1] scale: 0=uncertain, 1=very confident
        return proba, confidence

    def predict_signal(self, x) -> list[dict]:
        """High-level: returns list of signal dicts above confidence threshold."""
        proba, confidence = self.predict_with_confidence(x)
        results = []
        for i in range(len(proba)):
            if confidence[i] >= self.confidence_threshold:
                results.append({
                    "prediction": "up" if proba[i] > 0.5 else "down",
                    "probability": float(proba[i]),
                    "confidence": float(confidence[i]),
                })
            else:
                results.append({"prediction": "neutral", "probability": float(proba[i]),
                                 "confidence": float(confidence[i])})
        return results

    def train_epoch(self, loader, optimizer=None, criterion=None) -> dict:
        return {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, loader) -> EvalMetrics:
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

    def save(self, path: str, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights,
            "confidence_threshold": self.confidence_threshold,
            "gnn_weight": self.gnn_weight,
            **(metadata or {}),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        data = json.loads(Path(path).read_text())
        return cls(
            weights=data.get("weights"),
            confidence_threshold=data.get("confidence_threshold", 0.65),
            gnn_weight=data.get("gnn_weight", 0.0),
        )
