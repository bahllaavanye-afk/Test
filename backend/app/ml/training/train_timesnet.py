"""
TimesNet training entry point.
Mirrors train_lstm.py — same data pipeline, same trainer, different architecture.

TimesNet (Wu et al., ICLR 2023) uses FFT-detected periods to reshape 1D time series
into 2D for Conv2D processing — captures multi-scale periodicity simultaneously.

Usage:
    python -m app.ml.training.train_timesnet --symbol BTC/USDT --interval 1h --epochs 100

Free GPU:
    Upload OHLCV CSV to Kaggle → run this script on T4 → download .pt artifact
    Or use notebooks/train_timesnet.ipynb on Google Colab
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ml.features.engineer import engineer_features, create_sequences, add_labels
from app.ml.models.timesnet_model import TimesNetWrapper
from app.ml.training.trainer import ARTIFACTS_DIR
from app.utils.logging import logger


def build_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 60,
    batch_size: int = 256,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    df = engineer_features(df)
    df = add_labels(df, threshold=0.002)
    X, y = create_sequences(df, seq_len=seq_len)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    n = len(X_t)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
    val_ds = TensorDataset(X_t[n_train:n_train + n_val], y_t[n_train:n_train + n_val])
    test_ds = TensorDataset(X_t[n_train + n_val:], y_t[n_train + n_val:])

    n_features = X_t.shape[2]
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        n_features,
    )


async def train(
    ohlcv_df: pd.DataFrame,
    experiment_name: str = "timesnet_default",
    d_model: int = 64,
    d_ff: int = 128,
    n_layers: int = 3,
    top_k: int = 3,
    dropout: float = 0.1,
    seq_len: int = 60,
    max_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
) -> dict:
    train_loader, val_loader, test_loader, n_features = build_dataloaders(
        ohlcv_df, seq_len, batch_size
    )

    wrapper = TimesNetWrapper(
        input_size=n_features,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        top_k=top_k,
        dropout=dropout,
        lr=lr,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        train_metrics = wrapper.train_epoch(train_loader)
        val_metrics = wrapper.evaluate(val_loader)

        logger.info(
            "TimesNet epoch %d/%d — loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1, max_epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["val_loss"], val_metrics["val_accuracy"],
        )

        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            save_path = ARTIFACTS_DIR / experiment_name / "best_model.pt"
            wrapper.save(str(save_path), metadata={
                "input_size": n_features,
                "seq_len": seq_len,
                "d_model": d_model,
                "d_ff": d_ff,
                "n_layers": n_layers,
                "top_k": top_k,
                "epoch": epoch,
                "val_accuracy": best_val_acc,
            })
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Evaluate on test set
    test_metrics = wrapper.evaluate(test_loader)

    results = {
        "experiment": experiment_name,
        "model": "timesnet",
        "best_epoch": best_epoch,
        "val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_metrics["val_accuracy"], 4),
        "n_features": n_features,
        "artifact_path": str(ARTIFACTS_DIR / experiment_name / "best_model.pt"),
    }

    # Save results JSON
    results_path = ARTIFACTS_DIR / experiment_name / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("TimesNet training complete: %s", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TimesNet model")
    parser.add_argument("--csv", help="Path to OHLCV CSV file")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--experiment", default="timesnet_default")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
    else:
        # Minimal synthetic data for smoke test when no data provided
        import numpy as np
        dates = pd.date_range("2021-01-01", periods=2000, freq="1h")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "open": rng.uniform(30000, 70000, 2000),
            "high": rng.uniform(30000, 70000, 2000),
            "low": rng.uniform(30000, 70000, 2000),
            "close": rng.uniform(30000, 70000, 2000),
            "volume": rng.uniform(1e6, 1e9, 2000),
        }, index=dates)
        logger.warning("No CSV provided — using synthetic data for smoke test")

    asyncio.run(train(
        ohlcv_df=df,
        experiment_name=args.experiment,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        top_k=args.top_k,
        seq_len=args.seq_len,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    ))


if __name__ == "__main__":
    main()
