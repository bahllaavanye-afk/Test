"""LSTM training entry point. Can be run directly or via experiment config."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ml.features.engineer import engineer_features, create_sequences, add_labels
from app.ml.models.lstm import LSTMPredictor
from app.ml.training.trainer import train_with_lightning, ARTIFACTS_DIR
from app.utils.logging import logger


def _validate_build_params(
    df: pd.DataFrame | None,
    seq_len: int,
    batch_size: int,
    train_frac: float,
    val_frac: float,
) -> None:
    """Validate inputs for ``build_dataloaders``.

    Raises:
        ValueError: If any argument is invalid.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not (0.0 <= train_frac <= 1.0):
        raise ValueError("train_frac must be between 0 and 1.")
    if not (0.0 <= val_frac <= 1.0):
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac > 1.0:
        raise ValueError("train_frac + val_frac cannot exceed 1.0.")


def build_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 60,
    batch_size: int = 256,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create PyTorch DataLoaders for training, validation, and testing.

    Handles edge‑cases such as ``None`` inputs, empty dataframes,
    and ensures slicing does not exceed array bounds.
    """
    _validate_build_params(df, seq_len, batch_size, train_frac, val_frac)

    df = engineer_features(df)
    df = add_labels(df, threshold=0.002)

    X, y = create_sequences(df, seq_len=seq_len)

    # Guard against empty sequence generation (e.g., insufficient rows)
    if len(X) == 0 or len(y) == 0:
        raise ValueError(
            f"Sequence generation resulted in empty data. Ensure the input DataFrame has at least {seq_len} rows after feature engineering."
        )

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    n = len(X_t)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Ensure indices stay within bounds
    train_end = n_train
    val_end = n_train + n_val
    test_start = val_end

    train_ds = TensorDataset(X_t[:train_end], y_t[:train_end])
    val_ds = TensorDataset(X_t[train_end:val_end], y_t[train_end:val_end])
    test_ds = TensorDataset(X_t[test_start:], y_t[test_start:])

    n_features = X_t.shape[2]
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        n_features,
    )


async def train(
    ohlcv_df: pd.DataFrame,
    experiment_name: str = "lstm_default",
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    seq_len: int = 60,
    max_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict:
    """Train an LSTM model on the provided OHLCV DataFrame.

    Performs basic validation before invoking the training pipeline.
    """
    if ohlcv_df is None:
        raise ValueError("ohlcv_df cannot be None.")
    if ohlcv_df.empty:
        raise ValueError("ohlcv_df is empty; no data to train on.")

    train_loader, val_loader, test_loader, n_features = build_dataloaders(
        ohlcv_df, seq_len, batch_size
    )

    model = LSTMPredictor(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    results = train_with_lightning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name,
        max_epochs=max_epochs,
        lr=lr,
    )

    # Save final model
    save_path = ARTIFACTS_DIR / experiment_name / "final_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "seq_len": seq_len,
            "experiment": experiment_name,
        },
        str(save_path),
    )

    results["artifact_path"] = str(save_path)
    logger.info("LSTM training complete", **results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV with columns: open,high,low,close,volume")
    parser.add_argument("--name", default="lstm_run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    result = asyncio.run(
        train(
            df,
            experiment_name=args.name,
            max_epochs=args.epochs,
            hidden_size=args.hidden,
        )
    )
    print(json.dumps(result, indent=2))