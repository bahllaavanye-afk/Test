from __future__ import annotations

"""
LSTM model with self‑attention for binary time‑series direction prediction.

This module defines a bidirectional LSTM network augmented with a simple
self‑attention mechanism. The model is intended for classification tasks where
the target is a binary direction (up/down). It follows the architecture:

    Input → Bidirectional LSTM → Self‑Attention → LayerNorm →
    Linear → GELU → Dropout → Linear → Sigmoid

The model conforms to the ``AbstractModel`` interface used throughout the
codebase, providing ``train_epoch`` and ``evaluate`` helpers that work with
PyTorch data loaders.
"""

import logging
from typing import Iterable, Tuple, Dict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

import numpy as np
from sklearn.metrics import roc_auc_score

from app.ml.models.base_model import AbstractModel, EvalMetrics

logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    """
    Simple self‑attention layer.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the input feature vectors (typically ``hidden *
        directions`` from the LSTM).

    The layer learns a linear projection from the hidden dimension to a scalar
    attention score for each time step, then computes a weighted sum of the
    input sequence.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention‑weighted representation of the sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, hidden)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, hidden)`` representing the weighted
            aggregation over the time dimension.
        """
        if x.dim() != 3:
            raise ValueError(
                f"SelfAttention expected a 3‑D tensor (batch, seq_len, hidden), got shape {x.shape}"
            )
        scores = self.attention(x)                  # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)      # (batch, seq, 1)
        return (weights * x).sum(dim=1)             # (batch, hidden)


class LSTMPredictor(AbstractModel, nn.Module):
    """
    Bidirectional LSTM with self‑attention for binary classification.

    The model is compatible with the ``AbstractModel`` interface and can be
    trained/evaluated using standard PyTorch data loaders.

    Parameters
    ----------
    n_features : int, default 27
        Number of input features per time step.
    hidden_size : int, default 128
        Hidden size of the LSTM cells.
    num_layers : int, default 2
        Number of stacked LSTM layers.
    dropout : float, default 0.3
        Dropout probability applied after the linear head and between LSTM
        layers (if ``num_layers > 1``).
    bidirectional : bool, default True
        Whether to use a bidirectional LSTM.
    """

    model_type = "lstm"

    def __init__(
        self,
        n_features: int = 27,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMPredictor but is not installed.")
        nn.Module.__init__(self)
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        dirs = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.attention = SelfAttention(hidden_size * dirs)
        self.norm = nn.LayerNorm(hidden_size * dirs)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * dirs, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns
        -------
        torch.Tensor
            Logits tensor of shape ``(batch,)``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"LSTMPredictor.forward expected a 3‑D tensor (batch, seq_len, n_features), got shape {x.shape}"
            )
        try:
            out, _ = self.lstm(x)               # (batch, seq, hidden*dirs)
            ctx = self.attention(out)           # (batch, hidden*dirs)
            ctx = self.norm(ctx)
            return self.head(ctx).squeeze(-1)   # (batch,) logits
        except RuntimeError as e:
            logger.exception("Runtime error during forward pass")
            raise RuntimeError(f"Forward pass failed: {e}") from e

    def train_epoch(
        self,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Run a single training epoch.

        Parameters
        ----------
        loader : iterable of (features, labels)
            Data loader yielding batches of inputs and targets.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        criterion : torch.nn.Module
            Loss function (e.g., ``BCEWithLogitsLoss``).

        Returns
        -------
        dict
            Dictionary containing average ``loss`` and ``accuracy`` for the epoch.
        """
        self.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            try:
                optimizer.zero_grad()
                logits = self.forward(X)
                loss = criterion(logits, y.float())
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(y)
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == y.long()).sum().item()
                total += len(y)
            except (ValueError, RuntimeError) as e:
                logger.exception("Error during training batch")
                raise RuntimeError(f"Training epoch failed on a batch: {e}") from e
        if total == 0:
            raise RuntimeError("Training epoch encountered an empty loader.")
        return {"loss": total_loss / total, "accuracy": correct / total}

    def evaluate(self, loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> EvalMetrics:
        """
        Evaluate the model on a validation/test set.

        Parameters
        ----------
        loader : iterable of (features, labels)
            Data loader providing evaluation data.

        Returns
        -------
        EvalMetrics
            Structured metrics including accuracy, AUC, Sharpe (set to 0.0),
            and average loss.
        """
        self.eval()
        all_logits, all_labels = [], []
        total_loss, total = 0.0, 0
        criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            for X, y in loader:
                try:
                    logits = self.forward(X)
                    loss = criterion(logits, y.float())
                    total_loss += loss.item() * len(y)
                    all_logits.append(logits)
                    all_labels.append(y)
                    total += len(y)
                except (ValueError, RuntimeError) as e:
                    logger.exception("Error during evaluation batch")
                    raise RuntimeError(f"Evaluation failed on a batch: {e}") from e
        if total == 0:
            raise RuntimeError("Evaluation encountered an empty loader.")
        logits_cat = torch.cat(all_logits).numpy()
        labels_cat = torch.cat(all_labels).numpy()
        probs = 1 / (1 + np.exp(-logits_cat))
        preds = (probs > 0.5).astype(int)
        acc = (preds == labels_cat).mean()
        try:
            auc = float(roc_auc_score(labels_cat, probs))
        except ValueError as e:
            logger.warning("AUC calculation failed: %s. Defaulting to 0.5.", e)
            auc = 0.5
        return EvalMetrics(accuracy=float(acc), auc=auc, sharpe=0.0, loss=total_loss / total)