"""
Generic PyTorch Lightning Trainer wrapper with MLflow experiment tracking.
Supports LSTM, Transformer, and any nn.Module wrapped as a LightningModule.
GPU-maximized: mixed-precision, cudnn benchmark, multi-GPU, gradient accumulation,
gradient clipping, and optimized DataLoader kwargs.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.cuda.amp import GradScaler, autocast
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

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
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        pred = self(x).squeeze(-1)
        loss = self.criterion(pred, y.float())
        acc = ((torch.sigmoid(pred) > 0.5) == y.bool()).float().mean()
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
        except Exception as exc:
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


class Trainer:
    """
    GPU-maximized manual training loop.

    Features:
    - Mixed-precision (AMP) with GradScaler
    - cudnn.benchmark for faster convolutions
    - Multi-GPU via DataParallel
    - Gradient accumulation (accumulation_steps)
    - Gradient clipping (max_norm=1.0)
    - GPU memory logging every 10 epochs
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device | None = None,
        accumulation_steps: int = 1,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accumulation_steps = max(1, accumulation_steps)

        # Enable cudnn benchmark when CUDA is available for faster convolutions
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Wrap model with DataParallel when multiple GPUs are available
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

        # Mixed-precision scaler (no-op on CPU)
        self.scaler: GradScaler | None = GradScaler() if (torch.cuda.is_available() and HAS_AMP) else None

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    @staticmethod
    def get_dataloader_kwargs() -> dict:
        """
        Return optimized DataLoader kwargs for the current environment.

        Usage::

            loader = DataLoader(dataset, batch_size=256, **Trainer.get_dataloader_kwargs())
        """
        pin_memory = torch.cuda.is_available()
        num_workers = min(4, os.cpu_count() or 1)
        persistent_workers = num_workers > 0
        return {
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader, epoch: int = 0) -> float:
        """
        Run one epoch of training with AMP, gradient accumulation, and clipping.

        Returns the mean training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        self.optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # ── Forward pass (with AMP when available) ──────────────────
            if self.scaler is not None and HAS_AMP:
                with autocast():
                    pred = self.model(x).squeeze(-1)
                    loss = self.criterion(pred, y.float())
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                pred = self.model(x).squeeze(-1)
                loss = self.criterion(pred, y.float())
                loss = loss / self.accumulation_steps
                loss.backward()

            # ── Optimizer step every accumulation_steps batches ─────────
            step_num = batch_idx + 1
            if step_num % self.accumulation_steps == 0:
                if self.scaler is not None:
                    # Unscale before clipping so norms are on the correct scale
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps * len(x)
            n_samples += len(x)

        # Handle remaining batches not covered by the accumulation window
        remaining = len(loader) % self.accumulation_steps
        if remaining != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()

        # ── GPU memory logging every 10 epochs ──────────────────────────
        if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
            alloc_gb = torch.cuda.memory_allocated() / 1e9
            max_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.info(
                "GPU memory",
                epoch=epoch + 1,
                allocated_gb=round(alloc_gb, 3),
                max_allocated_gb=round(max_alloc_gb, 3),
            )

        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """
        Evaluate model on *loader*.

        Returns (mean_loss, mean_accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        n_samples = 0

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            pred = self.model(x).squeeze(-1)
            loss = self.criterion(pred, y.float())
            total_loss += loss.item() * len(x)
            total_correct += ((torch.sigmoid(pred) > 0.5) == y.bool()).float().sum().item()
            n_samples += len(x)

        n = max(n_samples, 1)
        return total_loss / n, total_correct / n


def _fallback_train(model, train_loader, val_loader, max_epochs, lr, patience):
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
    return {"val_loss": best_val_loss, "val_acc": val_acc, "best_model_path": "", "epochs_trained": epoch + 1}
