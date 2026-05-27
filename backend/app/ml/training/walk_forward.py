"""
Walk-forward cross-validation for ML models.
Prevents temporal leakage by training only on past data at each fold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from app.ml.features.engineer import engineer_features, create_sequences, add_labels
    HAS_FEATURES = True
except ImportError:
    HAS_FEATURES = False

from app.utils.logging import logger


@dataclass
class WalkForwardResult:
    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    val_sharpe: float
    val_accuracy: float
    val_loss: float
    n_train_samples: int
    n_val_samples: int


def _build_tensors(
    df: pd.DataFrame,
    seq_len: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Engineer features and create (X, y) tensors from a price DataFrame."""
    if not HAS_FEATURES:
        raise ImportError("app.ml.features.engineer not available")
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed")

    feat_df = engineer_features(df)
    feat_df = add_labels(feat_df, threshold=0.002)
    X, y = create_sequences(feat_df, seq_len=seq_len)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return X_t, y_t


def _train_one_fold(
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    max_epochs: int,
    lr: float = 1e-3,
    patience: int = 10,
) -> tuple[float, float]:
    """
    Minimal training loop for one walk-forward fold.
    Returns (val_loss, val_accuracy).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = math.inf
    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch).squeeze(-1)
            loss = criterion(pred, y_batch.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch).squeeze(-1)
                val_losses.append(criterion(pred, y_batch.float()).item())
                correct = ((torch.sigmoid(pred) > 0.5) == y_batch.bool()).float().mean().item()
                val_accs.append(correct)

        val_loss = float(np.mean(val_losses)) if val_losses else math.inf
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.debug("Early stop", fold_epoch=epoch, val_loss=round(val_loss, 4))
                break

    return best_val_loss, best_val_acc


def _sharpe_from_accuracy(val_accuracy: float) -> float:
    """
    Heuristic: map classification accuracy to a Sharpe proxy.
    A perfectly random model (50%) → Sharpe 0; 70% acc → ~2.0 Sharpe.
    This is used only when no equity-level evaluation is done in the fold.
    """
    # linear rescale: accuracy in [0.5, 1.0] → sharpe in [0, 3]
    clamped = max(0.5, min(1.0, val_accuracy))
    return round((clamped - 0.5) / 0.5 * 3.0, 4)


def walk_forward_validate(
    df: pd.DataFrame,
    model_factory: Callable[[], "nn.Module"],
    n_folds: int = 5,
    train_window_days: int = 365,
    val_window_days: int = 90,
    seq_len: int = 60,
    batch_size: int = 256,
    max_epochs: int = 50,
) -> list[WalkForwardResult]:
    """
    Walk-forward validation for any PyTorch model.

    Each fold uses a non-overlapping expanding window strategy:
    - Fold 0: train on [0, train_window], val on [train_window, train_window + val_window]
    - Fold 1: train on [0, train_window + val_window], val on [train_window + val_window, ...]
    - ...

    Parameters
    ----------
    df              : OHLCV DataFrame with DatetimeIndex.
    model_factory   : Callable with no args that returns a fresh nn.Module per fold.
    n_folds         : Number of walk-forward folds.
    train_window_days: Initial training window in calendar days.
    val_window_days : Validation window in calendar days.
    seq_len         : Sequence length for feature engineering.
    batch_size      : DataLoader batch size.
    max_epochs      : Max epochs per fold.

    Returns
    -------
    List of WalkForwardResult, one per fold.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch must be installed to use walk_forward_validate")

    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    results: list[WalkForwardResult] = []

    train_delta = timedelta(days=train_window_days)
    val_delta = timedelta(days=val_window_days)

    start_date = df.index[0]

    for fold in range(n_folds):
        # Expanding train window — each fold adds one val window of history
        train_end = start_date + train_delta + fold * val_delta
        val_start = train_end
        val_end = val_start + val_delta

        if val_end > df.index[-1]:
            logger.info("Walk-forward: no more data for fold", fold=fold)
            break

        train_df = df.loc[start_date:train_end]
        val_df = df.loc[val_start:val_end]

        if len(train_df) < seq_len + 10 or len(val_df) < seq_len + 5:
            logger.warning("Insufficient data for fold", fold=fold,
                           train_bars=len(train_df), val_bars=len(val_df))
            continue

        try:
            X_train, y_train = _build_tensors(train_df, seq_len)
            X_val, y_val = _build_tensors(val_df, seq_len)
        except Exception as exc:
            logger.warning("Feature engineering failed", fold=fold, error=str(exc))
            results.append(WalkForwardResult(
                fold_id=fold,
                train_start=str(train_df.index[0].date()),
                train_end=str(train_df.index[-1].date()),
                val_start=str(val_df.index[0].date()),
                val_end=str(val_df.index[-1].date()),
                val_sharpe=0.0,
                val_accuracy=0.0,
                val_loss=999.0,
                n_train_samples=0,
                n_val_samples=0,
            ))
            continue

        n_train = len(X_train)
        n_val = len(X_val)

        if n_train == 0 or n_val == 0:
            logger.warning("Empty tensors after sequencing", fold=fold)
            continue

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = model_factory()
        val_loss, val_acc = _train_one_fold(
            model, train_loader, val_loader, max_epochs=max_epochs
        )
        val_sharpe = _sharpe_from_accuracy(val_acc)

        logger.info(
            "Walk-forward fold complete",
            fold=fold,
            train_bars=len(train_df),
            val_bars=len(val_df),
            val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 4),
            val_sharpe=val_sharpe,
        )

        results.append(WalkForwardResult(
            fold_id=fold,
            train_start=str(train_df.index[0].date()),
            train_end=str(train_df.index[-1].date()),
            val_start=str(val_df.index[0].date()),
            val_end=str(val_df.index[-1].date()),
            val_sharpe=val_sharpe,
            val_accuracy=round(val_acc, 4),
            val_loss=round(val_loss, 4),
            n_train_samples=n_train,
            n_val_samples=n_val,
        ))

    return results
