from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import torch
import numpy as np


@dataclass
class EvalMetrics:
    accuracy: float
    auc: float
    sharpe: float
    loss: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None


class AbstractModel(ABC):
    """
    Base class for all QuantEdge ML models.
    Every model must implement: forward, train_epoch, evaluate.
    Save/load are provided by default.
    """
    model_type: str = "base"

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward pass. Returns logits or probabilities."""

    @abstractmethod
    def train_epoch(self, loader, optimizer, criterion) -> dict:
        """Train for one epoch. Returns dict with loss, acc, etc."""

    @abstractmethod
    def evaluate(self, loader) -> EvalMetrics:
        """Evaluate model on a DataLoader."""

    def save(self, path: str, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict() if hasattr(self, "state_dict") else {},
            "model_type": self.model_type,
            "metadata": metadata or {},
        }, path)
        if metadata:
            meta_path = Path(path).with_suffix(".json")
            meta_path.write_text(json.dumps(metadata, default=str, indent=2))

    @classmethod
    def load(cls, path: str) -> "AbstractModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**checkpoint.get("metadata", {}).get("init_kwargs", {}))
        if checkpoint["state_dict"] and hasattr(model, "load_state_dict"):
            model.load_state_dict(checkpoint["state_dict"])
        return model

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Returns probability of class 1 (up)."""
        self.eval() if hasattr(self, "eval") else None
        with torch.no_grad():
            logits = self.forward(x)
            if logits.shape[-1] == 1 or logits.dim() == 1:
                return torch.sigmoid(logits).numpy().flatten()
            return torch.softmax(logits, dim=-1)[:, 1].numpy()
