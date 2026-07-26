"""
Generic PyTorch Lightning Trainer wrapper with MLflow experiment tracking.
Supports LSTM, Transformer, and any nn.Module wrapped as a LightningModule.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import MLFlowLogger
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Optional torchmetrics for richer validation metrics
try:
    from torchmetrics import AUROC, Precision, Recall
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

from app.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).parents[4] / "models_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class TradingLightningModule(L.LightningModule if HAS_LIGHTNING else object):
    """Wraps any nn.Module for PyTorch Lightning training with additional signal‑quality metrics."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
        if HAS_LIGHTNING:
            super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.BCEWithLogitsLoss()

        # Initialize optional metrics for validation
        if HAS_TORCHMETRICS:
            self.val_auc = AUROC(pos_label=1)
            self.val_precision = Precision()
            self.val_recall = Recall()
        else:
            self.val_auc = self.val_precision = self.val_recall = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Common logic for train/validation steps."""
        x, y = batch
        pred = self(x).squeeze(-1)
        loss = self.criterion(pred, y.float())
        prob = torch.sigmoid(pred)
        acc = ((prob > 0.5) == y.bool()).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        if stage == "val" and HAS_TORCHMETRICS:
            self.val_auc.update(prob, y)
            self.val_precision.update(prob, y)
            self.val_recall.update(prob, y)

        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Log additional validation metrics after each epoch if available."""
        if HAS_TORCHMETRICS:
            self.log("val_auc", self.val_auc.compute(), prog_bar=True)
            self.log("val_precision", self.val_precision.compute(), prog_bar=True)
            self.log("val_recall", self.val_recall.compute(), prog_bar=True)
            self.val_auc.reset()
            self.val_precision.reset()
            self.val_recall.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
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
) -> dict[str, Any]:
    """
    Train model with PyTorch Lightning + optional MLflow logging.
    Returns a dict with validation loss, accuracy, additional metrics and checkpoint info.
    Falls back to a manual training loop if Lightning is not installed.
    """
    # Basic sanity checks for data loaders
    if train_loader is None or len(train_loader) == 0:
        raise ValueError("train_loader must contain at least one batch")
    if val_loader is None or len(val_loader) == 0:
        raise ValueError("val_loader must contain at least one batch")

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
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        min_delta=1e-4,  # require a minimal improvement to avoid premature stopping
    )
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
        "val_auc": float(trainer.callback_metrics.get("val_auc", 0)),
        "val_precision": float(trainer.callback_metrics.get("val_precision", 0)),
        "val_recall": float(trainer.callback_metrics.get("val_recall", 0)),
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
) -> dict[str, Any]:
    """Minimal training loop when Lightning is unavailable."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

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
        val_losses, val_accs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze(-1)
                val_losses.append(criterion(pred, y.float()).item())
                prob = torch.sigmoid(pred)
                val_accs.append(((prob > 0.5) == y.bool()).float().mean().item())

        val_loss = sum(val_losses) / len(val_losses) if val_losses else 999
        val_acc = sum(val_accs) / len(val_accs) if val_accs else 0

        if val_loss < best_val_loss - 1e-4:  # enforce a minimal improvement
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
        "val_auc": 0.0,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "best_model_path": "",
        "epochs_trained": epoch + 1,
    }