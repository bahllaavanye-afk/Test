"""LSTM training entry point. Can be run directly or via experiment config."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ml.features.engineer import engineer_features, create_sequences, add_labels
from app.ml.models.lstm import LSTMPredictor
from app.ml.training.trainer import train_with_lightning, ARTIFACTS_DIR
from app.utils.logging import logger


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the DataFrame is suitable for training."""
    if df is None:
        raise ValueError("Input DataFrame is None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df)}.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    # Basic column check – adjust as needed for feature engineering expectations
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _validate_split_fractions(train_frac: float, val_frac: float) -> None:
    """Ensure split fractions are within (0,1) and sum to <= 1."""
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be between 0 and 1 (exclusive).")
    if not (0 < val_frac < 1):
        raise ValueError("val_frac must be between 0 and 1 (exclusive).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1.")


def build_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 60,
    batch_size: int = 256,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Construct train/validation/test DataLoaders from raw OHLCV data.

    Handles edge cases such as None inputs, empty DataFrames, and invalid
    sequence lengths or batch sizes.
    """
    _validate_dataframe(df)
    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    _validate_split_fractions(train_frac, val_frac)

    # Feature engineering
    df = engineer_features(df)
    df = add_labels(df, threshold=0.002)

    # Sequence creation
    X, y = create_sequences(df, seq_len=seq_len)
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No sequences generated; check seq_len and input data size.")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    n = len(X_t)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Guard against off‑by‑one indexing issues
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
    val_ds = TensorDataset(
        X_t[n_train : n_train + n_val],
        y_t[n_train : n_train + n_val],
    )
    test_ds = TensorDataset(
        X_t[n_train + n_val :],
        y_t[n_train + n_val :],
    )

    n_features = X_t.shape[2] if X_t.ndim == 3 else 1

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
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
    """Orchestrate LSTM training with robust edge‑case handling."""
    try:
        _validate_dataframe(ohlcv_df)
        if seq_len <= 0:
            raise ValueError("seq_len must be a positive integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer.")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0.")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer.")
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")

        train_loader, val_loader, test_loader, n_features = build_dataloaders(
            ohlcv_df, seq_len=seq_len, batch_size=batch_size
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

    except Exception as exc:  # pragma: no cover
        logger.error("LSTM training failed", error=str(exc))
        return {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to OHLCV CSV with columns: open,high,low,close,volume",
    )
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