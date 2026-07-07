from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import time
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """
    Container for evaluation metrics produced by a model.

    Attributes
    ----------
    accuracy : float
        Classification accuracy.
    auc : float
        Area under the ROC curve.
    sharpe : float
        Sharpe ratio of the model's predictions.
    loss : float | None, default None
        Optional loss value.
    f1 : float | None, default None
        Optional F1 score.
    precision : float | None, default None
        Optional precision metric.
    recall : float | None, default None
        Optional recall metric.
    """
    accuracy: float
    auc: float
    sharpe: float
    loss: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None


class AbstractModel(ABC):
    """
    Abstract base class for all QuantEdge machine‑learning models.

    Sub‑classes must implement the core training and inference methods.
    The class provides generic ``save`` and ``load`` utilities that handle
    PyTorch state dictionaries when the library is available.
    """

    model_type: str = "base"

    @abstractmethod
    def forward(self, x: Any) -> Any:  # type: ignore[override]
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        x : Any
            Input data (typically a torch.Tensor or numpy array).

        Returns
        -------
        Any
            Model output – logits or probabilities depending on the implementation.
        """

    @abstractmethod
    def train_epoch(
        self,
        loader: Any,
        optimizer: Any,
        criterion: Any,
    ) -> Dict[str, Any]:
        """
        Execute a single training epoch.

        Parameters
        ----------
        loader : Any
            Iterable data loader yielding training batches.
        optimizer : Any
            Optimizer instance used for parameter updates.
        criterion : Any
            Loss function used to compute the training loss.

        Returns
        -------
        dict
            Dictionary containing training statistics such as loss and accuracy.
        """

    @abstractmethod
    def evaluate(self, loader: Any) -> EvalMetrics:
        """
        Evaluate the model on a validation or test set.

        Parameters
        ----------
        loader : Any
            Iterable data loader yielding evaluation batches.

        Returns
        -------
        EvalMetrics
            Aggregated evaluation metrics.
        """

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Serialize the model to disk.

        Parameters
        ----------
        path : str
            Destination file path for the checkpoint.
        metadata : dict | None, optional
            Optional dictionary of auxiliary information to store alongside the model.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if TORCH_AVAILABLE:
            import torch as _torch
            _torch.save(
                {
                    "state_dict": self.state_dict() if hasattr(self, "state_dict") else {},  # type: ignore[attr-defined]
                    "model_type": self.model_type,
                    "metadata": metadata or {},
                },
                path,
            )
        if metadata:
            meta_path = Path(path).with_suffix(".json")
            meta_path.write_text(json.dumps(metadata, default=str, indent=2))

    @classmethod
    def load(cls, path: str) -> "AbstractModel":
        """
        Load a model checkpoint from disk.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.

        Returns
        -------
        AbstractModel
            An instance of the model class with loaded parameters.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required to load model checkpoints")
        import torch as _torch
        checkpoint = _torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**checkpoint.get("metadata", {}).get("init_kwargs", {}))
        if checkpoint["state_dict"] and hasattr(model, "load_state_dict"):
            model.load_state_dict(checkpoint["state_dict"])  # type: ignore[attr-defined]
        return model

    def predict_proba(self, x: Any) -> np.ndarray:
        """
        Generate class‑1 probabilities for the provided inputs.

        Parameters
        ----------
        x : Any
            Input data suitable for the model's forward method.

        Returns
        -------
        np.ndarray
            Array of probabilities for the positive class.
        """
        start_time = time.time()
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for inference")
        import torch as _torch
        if hasattr(self, "eval"):
            self.eval()  # type: ignore[attr-defined]
        with _torch.no_grad():
            logits = self.forward(x)
            if logits.shape[-1] == 1 or logits.dim() == 1:
                probs = _torch.sigmoid(logits).numpy().flatten()
            else:
                probs = _torch.softmax(logits, dim=-1)[:, 1].numpy()
        exec_time = time.time() - start_time

        # Determine signal count safely
        try:
            signal_count = int(x.shape[0])
        except Exception:
            try:
                signal_count = len(x)
            except Exception:
                signal_count = 0

        # Log structured metrics; P&L is not known at inference time, default to 0.0
        self.log_metrics(signal_count=signal_count, exec_time=exec_time, pnl=0.0)

        return probs

    def log_metrics(self, signal_count: int, exec_time: float, pnl: float) -> None:
        """
        Log key runtime metrics in a structured format.

        Parameters
        ----------
        signal_count : int
            Number of signals (e.g., rows) processed.
        exec_time : float
            Execution time in seconds for the operation.
        pnl : float
            Profit & loss associated with the processed signals.
        """
        logger.info(
            "model_metrics",
            extra={
                "signal_count": signal_count,
                "execution_time_sec": exec_time,
                "pnl": pnl,
                "model_type": getattr(self, "model_type", "unknown"),
            },
        )