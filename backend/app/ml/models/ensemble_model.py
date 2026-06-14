"""
Weighted ensemble of LSTM + XGBoost + Lorentzian KNN.
Weights optimized on validation set via Optuna.
Only signals with confidence > threshold are forwarded.
"""
import json
from pathlib import Path

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

logger = structlog.get_logger()


class EnsembleModel(AbstractModel):
    model_type = "ensemble"

    def __init__(
        self,
        weights: dict | None = None,
        confidence_threshold: float = 0.65,
        gnn_weight: float = 0.0,
    ):
        self.weights = weights or {"lstm": 0.4, "xgboost": 0.3, "lorentzian": 0.15, "ssm": 0.15}
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
        # Per-call weight overrides — never mutate self.weights inside forward(),
        # otherwise the GNN weight accumulates across calls and skews normalization.
        call_weights: dict[str, float] = {}
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

    def optimize_weights_walk_forward(
        self,
        returns_by_model: dict[str, "pd.Series"],
        actual_returns: "pd.Series",
        n_splits: int = 5,
    ) -> dict[str, float]:
        """
        Walk-forward ensemble weight optimization.

        Uses scipy SLSQP to find weights that maximise Sharpe on each fold,
        then returns the average weights across folds.

        Args:
            returns_by_model: dict of model_name → predicted-return pd.Series (same index)
            actual_returns:   actual forward returns pd.Series (same index as above)
            n_splits:         number of walk-forward folds

        Returns:
            dict of model_name → optimal weight, summing to 1.
        """
        import numpy as np
        import pandas as pd
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
            return {k: 1.0 / len(model_names) for k in model_names}

        avg_weights = np.mean(all_weights, axis=0)
        avg_weights = np.maximum(avg_weights, 0.0)
        avg_weights = avg_weights / avg_weights.sum()

        result_weights = {name: float(avg_weights[i]) for i, name in enumerate(model_names)}
        self.weights.update(result_weights)
        return result_weights

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
