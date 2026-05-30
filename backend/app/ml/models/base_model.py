from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


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

    All torch usage is guarded by TORCH_AVAILABLE so that the class can be
    imported on environments without PyTorch (e.g. Render free tier).
    """
    model_type: str = "base"

    @abstractmethod
    def forward(self, x):  # type: ignore[override]
        """Inference forward pass. Returns logits or probabilities."""

    @abstractmethod
    def train_epoch(self, loader, optimizer, criterion) -> dict:
        """Train for one epoch. Returns dict with loss, acc, etc."""

    @abstractmethod
    def evaluate(self, loader) -> EvalMetrics:
        """Evaluate model on a DataLoader."""

    def save(self, path: str, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if TORCH_AVAILABLE:
            import torch as _torch
            _torch.save({
                "state_dict": self.state_dict() if hasattr(self, "state_dict") else {},  # type: ignore[attr-defined]
                "model_type": self.model_type,
                "metadata": metadata or {},
            }, path)
        if metadata:
            meta_path = Path(path).with_suffix(".json")
            meta_path.write_text(json.dumps(metadata, default=str, indent=2))

    @classmethod
    def load(cls, path: str) -> "AbstractModel":
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required to load model checkpoints")
        import torch as _torch
        checkpoint = _torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**checkpoint.get("metadata", {}).get("init_kwargs", {}))
        if checkpoint["state_dict"] and hasattr(model, "load_state_dict"):
            model.load_state_dict(checkpoint["state_dict"])  # type: ignore[attr-defined]
        return model

    def predict_proba(self, x) -> np.ndarray:
        """Returns probability of class 1 (up)."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for inference")
        import torch as _torch
        if hasattr(self, "eval"):
            self.eval()  # type: ignore[attr-defined]
        with _torch.no_grad():
            logits = self.forward(x)
            if logits.shape[-1] == 1 or logits.dim() == 1:
                return _torch.sigmoid(logits).numpy().flatten()
            return _torch.softmax(logits, dim=-1)[:, 1].numpy()
