"""
Generic PyTorch Lightning Trainer wrapper with MLflow experiment tracking.
Supports LSTM, Transformer, and any nn.Module wrapped as a LightningModule.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

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

from app.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).parents[4] / "models_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class TradingLightningModule(L.LightningModule if HAS_LIGHTNING else object):
    """Wraps any nn.Module for PyTorch Lightning training."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
        if HAS_LIGHTNING:
            super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        pred = self(x).squeeze(-1)
        loss = self.criterion(pred, y.float())
        acc = ((pred > 0.5) == y.bool()).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
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
    Train model with PyTorch Lightning + MLflow logging.
    Returns dict with val_loss, val_acc, best_checkpoint_path.
    Falls back to manual training loop if Lightning not installed.
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
        except Exception:
            pass

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


def _fallback_train(model, train_loader, val_loader, max_epochs, lr, patience):
    """Minimal training loop when Lightning is unavailable."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
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
                val_accs.append(((pred > 0.5) == y.bool()).float().mean().item())

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
    return {"val_loss": best_val_loss, "val_acc": val_acc, "best_model_path": "", "epochs_trained": epoch + 1}
