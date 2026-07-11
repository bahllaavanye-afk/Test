from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Callable, Dict, Optional, Union

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

    # Default signal parameters – can be overridden by subclasses
    ENTRY_THRESHOLD: float = 0.60          # probability above which an entry is considered
    CONFIRMATION_WINDOW: int = 5           # number of recent predictions to smooth
    CONFIRMATION_STD_MAX: float = 0.10     # max std dev allowed for confirmation
    EXIT_PROFIT_TARGET: float = 0.02       # 2% profit target
    EXIT_STOP_LOSS: float = 0.01           # 1% stop‑loss

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

    # ------------------------------------------------------------------
    # Persistence utilities
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
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
    # Strategy signal logic
    # ------------------------------------------------------------------
    def _smooth_predictions(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply a simple moving‑average smoothing to a probability series.

        Parameters
        ----------
        probs : np.ndarray
            Raw probability predictions.

        Returns
        -------
        np.ndarray
            Smoothed probabilities.
        """
        if self.CONFIRMATION_WINDOW <= 1:
            return probs
        cumsum = np.cumsum(np.insert(probs, 0, 0))
        smoothed = (cumsum[self.CONFIRMATION_WINDOW:] -
                    cumsum[:-self.CONFIRMATION_WINDOW]) / self.CONFIRMATION_WINDOW
        # Pad to original length
        pad = np.full(self.CONFIRMATION_WINDOW - 1, smoothed[0])
        return np.concatenate([pad, smoothed])

    def _passes_confirmation(self, probs: np.ndarray) -> bool:
        """
        Determine whether the recent probability window is stable enough
        to trust an entry signal.

        Parameters
        ----------
        probs : np.ndarray
            Probability series for the confirmation window.

        Returns
        -------
        bool
            True if standard deviation is below the configured maximum.
        """
        if len(probs) < self.CONFIRMATION_WINDOW:
            return False
        recent = probs[-self.CONFIRMATION_WINDOW :]
        return float(np.std(recent)) <= self.CONFIRMATION_STD_MAX

    def generate_signal(
        self,
        x: Any,
        price_series: Optional[np.ndarray] = None,
        *,
        entry_threshold: Optional[float] = None,
        exit_profit_target: Optional[float] = None,
        exit_stop_loss: Optional[float] = None,
        confirmation_filter: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> int:
        """
        Produce a trading signal based on model probabilities and optional
        confirmation filters.

        The method implements a tightened entry condition:
        * Probability must exceed ``entry_threshold`` (default ``ENTRY_THRESHOLD``).
        * Recent predictions must be stable (low standard deviation).
        * An optional user‑provided ``confirmation_filter`` can impose
          additional domain‑specific checks.

        Exit logic is based on simple profit‑target and stop‑loss thresholds
        applied to the supplied ``price_series`` when a position is open.

        Parameters
        ----------
        x : Any
            Input features for the model.
        price_series : np.ndarray | None, optional
            Historical price series aligned with ``x``. Required for exit
            evaluation; if omitted, only entry logic is applied.
        entry_threshold : float | None, optional
            Override for the entry probability threshold.
        exit_profit_target : float | None, optional
            Override for the profit‑target percentage.
        exit_stop_loss : float | None, optional
            Override for the stop‑loss percentage.
        confirmation_filter : Callable[[np.ndarray], bool] | None, optional
            Custom callable receiving the recent probability window and
            returning ``True`` if the signal passes additional checks.

        Returns
        -------
        int
            ``1`` for a long entry, ``-1`` for a short entry, ``0`` for no signal
            or an exit condition.
        """
        # Resolve thresholds
        thr = entry_threshold if entry_threshold is not None else self.ENTRY_THRESHOLD
        profit_target = (
            exit_profit_target
            if exit_profit_target is not None
            else self.EXIT_PROFIT_TARGET
        )
        stop_loss = (
            exit_stop_loss if exit_stop_loss is not None else self.EXIT_STOP_LOSS
        )

        # Predict probabilities
        probs = self.predict_proba(x)

        # ------------------------------------------------------------------
        # Entry evaluation
        # ------------------------------------------------------------------
        if probs[-1] >= thr:
            # Apply smoothing for confirmation
            smoothed = self._smooth_predictions(probs)
            if self._passes_confirmation(smoothed):
                # Custom user filter
                if confirmation_filter is None or confirmation_filter(smoothed):
                    return 1  # long entry (short logic can be added by subclass)

        # ------------------------------------------------------------------
        # Exit evaluation (only if a position is presumed open)
        # ------------------------------------------------------------------
        if price_series is not None and len(price_series) >= 2:
            # Simple exit based on last entry price assumption:
            # Assume entry price was the price at the time of the last signal.
            entry_price = price_series[-2]
            current_price = price_series[-1]
            ret = (current_price - entry_price) / entry_price

            if ret >= profit_target or ret <= -stop_loss:
                return 0  # signal to exit position

        return 0  # default: no action

    # ------------------------------------------------------------------
    # Optional hooks for subclasses
    # ------------------------------------------------------------------
    def get_entry_threshold(self) -> float:
        """
        Return the current entry probability threshold.
        Sub‑classes may override to provide dynamic thresholds.
        """
        return self.ENTRY_THRESHOLD

    def get_exit_parameters(self) -> tuple[float, float]:
        """
        Return the profit‑target and stop‑loss percentages.
        """
        return self.EXIT_PROFIT_TARGET, self.EXIT_STOP_LOSS