from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, Optional, Union, Sequence

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


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

    # ------------------------------------------------------------------
    # Signal generation utilities
    # ------------------------------------------------------------------
    def generate_signal(
        self,
        x: Any,
        entry_thresh: float = 0.6,
        exit_thresh: float = 0.4,
        confirmation_window: int = 3,
        confirmation_std: float = 0.02,
    ) -> np.ndarray:
        """
        Produce trading signals (1 for long, -1 for short, 0 for flat) based on model
        probabilities with tightened entry conditions and confirmation filters.

        The logic follows:
        * **Entry** – probability > ``entry_thresh`` *and* the recent ``confirmation_window``
          probabilities are consistently above the threshold with limited volatility
          (standard deviation < ``confirmation_std``).
        * **Exit** – probability falls below ``exit_thresh`` *or* the confirmation filter
          fails (high volatility or probability crossing back under the entry threshold).

        Parameters
        ----------
        x : Any
            Input data passed to ``predict_proba``.
        entry_thresh : float, default 0.6
            Minimum probability required to consider opening a position.
        exit_thresh : float, default 0.4
            Probability below which an existing position is closed.
        confirmation_window : int, default 3
            Number of recent observations used for confirmation.
        confirmation_std : float, default 0.02
            Maximum allowed standard deviation within the confirmation window.

        Returns
        -------
        np.ndarray
            Array of signals aligned with the input order: 1 (long), -1 (short), 0 (flat).
        """
        probs = self.predict_proba(x)
        if probs.ndim != 1:
            raise ValueError("predict_proba must return a 1‑D array of probabilities")
        signals = np.zeros_like(probs, dtype=int)

        # Helper: confirm that a slice satisfies entry criteria
        def _confirm(slice_: np.ndarray) -> bool:
            return slice_.mean() > entry_thresh and slice_.std() <= confirmation_std

        for idx in range(len(probs)):
            # Entry condition
            if probs[idx] > entry_thresh:
                start = max(0, idx - confirmation_window + 1)
                window = probs[start : idx + 1]
                if len(window) == confirmation_window and _confirm(window):
                    signals[idx] = 1  # long signal (could be extended to short logic)
                    continue

            # Exit condition for an open position
            if signals[idx - 1] != 0 if idx > 0 else False:
                # if probability drops below exit threshold or confirmation fails
                if probs[idx] < exit_thresh:
                    signals[idx] = 0
                else:
                    # re‑evaluate confirmation; lose position if volatility spikes
                    start = max(0, idx - confirmation_window + 1)
                    window = probs[start : idx + 1]
                    if len(window) == confirmation_window and not _confirm(window):
                        signals[idx] = 0
                    else:
                        # maintain previous signal
                        signals[idx] = signals[idx - 1]

        return signals