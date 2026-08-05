"""
Generic PyTorch Lightning Trainer wrapper with MLflow experiment tracking.
Supports LSTM, Transformer, and any nn.Module wrapped as a LightningModule.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import MLFlowLogger

    HAS_LIGHTNING = True
except ImportError:  # pragma: no cover
    HAS_LIGHTNING = False

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    HAS_MLFLOW = False

from app.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).parents[4] / "models_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class TradingLightningModule(L.LightningModule if HAS_LIGHTNING else object):
    """LightningModule wrapper for an arbitrary ``nn.Module``.

    This class adapts a regular PyTorch model to the Lightning training
    interface, adding a binary‑cross‑entropy loss and simple accuracy metric.
    It is deliberately lightweight and can be used with any model that
    accepts a tensor input and returns a tensor output.

    Args:
        model: The underlying ``nn.Module`` to be trained.
        lr: Learning rate for the optimizer. Defaults to ``1e-3``.
        weight_decay: Weight decay (L2 regularisation) for the optimizer.
            Defaults to ``1e-4``.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> None:
        if HAS_LIGHTNING:
            super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass delegated to the wrapped model."""
        return self.model(x)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Common logic for training and validation steps.

        Computes loss and binary accuracy, logs both metrics, and returns the loss.

        Args:
            batch: Tuple ``(features, targets)`` where both are tensors.
            stage: Identifier used for logging (e.g., ``"train"`` or ``"val"``).

        Returns:
            The computed loss tensor.
        """
        x, y = batch
        pred = self(x).squeeze(-1)
        loss = self.criterion(pred, y.float())
        acc = ((torch.sigmoid(pred) > 0.5) == y.bool()).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning training step."""
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning validation step."""
        return self._step(batch, "val")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning‑rate scheduler for Lightning."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train_with_lightning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_name: str,
    max_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    mlflow_uri: str = "mlruns",
) -> Dict[str, Any]:
    """
    Train a model using PyTorch Lightning with optional MLflow tracking.

    The function prefers the Lightning training loop; if Lightning is not
    available it falls back to a minimal manual training loop.

    Args:
        model: The ``nn.Module`` to be trained.
        train_loader: DataLoader providing the training dataset.
        val_loader: DataLoader providing the validation dataset.
        experiment_name: Name of the MLflow experiment (also used for artifact storage).
        max_epochs: Maximum number of epochs to train. Defaults to ``100``.
        patience: Early‑stopping patience based on validation loss. Defaults to ``10``.
        lr: Learning rate for the optimizer. Defaults to ``1e-3``.
        mlflow_uri: URI for the MLflow tracking server. Defaults to ``"mlruns"``.

    Returns:
        A dictionary containing ``val_loss``, ``val_acc``, ``best_model_path``,
        and ``epochs_trained``.
    """
    if not HAS_LIGHTNING:
        logger.warning("PyTorch Lightning not installed — using fallback training loop")
        return _fallback_train(model, train_loader, val_loader, max_epochs, lr, patience)

    mlflow_logger = None
    if HAS_MLFLOW:
        try:
            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=mlflow_uri,
                run_name=experiment_name,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("MLflow logger init failed — proceeding without tracking", error=str(exc))

    lightning_module = TradingLightningModule(model, lr=lr)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ARTIFACTS_DIR / experiment_name),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=mlflow_logger,
        enable_progress_bar=False,
        log_every_n_steps=1,
        accelerator="auto",
        devices=1,
    )

    trainer.fit(lightning_module, train_loader, val_loader)

    results = {
        "val_loss": float(trainer.callback_metrics.get("val_loss", 999)),
        "val_acc": float(trainer.callback_metrics.get("val_acc", 0)),
        "best_model_path": checkpoint_cb.best_model_path,
        "epochs_trained": trainer.current_epoch,
    }
    logger.info("Lightning training complete", experiment=experiment_name, **results)
    return results


def _fallback_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    lr: float,
    patience: int,
) -> Dict[str, Any]:
    """
    Minimal training loop used when PyTorch Lightning is unavailable.

    This fallback implements a simple epoch loop with early stopping based on
    validation loss. It mirrors the essential behavior of the Lightning trainer
    without requiring the Lightning dependency.

    Args:
        model: The ``nn.Module`` to train.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        max_epochs: Maximum number of epochs to run.
        lr: Learning rate for the AdamW optimizer.
        patience: Number of consecutive epochs without improvement before stopping.

    Returns:
        A dictionary with keys ``val_loss``, ``val_acc``, ``best_model_path`` (empty
        string in this mode), and ``epochs_trained``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    patience_count = 0
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses: List[float] = []
        val_accs: List[float] = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze(-1)
                val_losses.append(criterion(pred, y.float()).item())
                val_accs.append(((torch.sigmoid(pred) > 0.5) == y.bool()).float().mean().item())

        val_loss = sum(val_losses) / len(val_losses) if val_losses else 999
        val_acc = sum(val_accs) / len(val_accs) if val_accs else 0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return {
        "val_loss": best_val_loss,
        "val_acc": val_acc,
        "best_model_path": "",
        "epochs_trained": epoch + 1,
    }