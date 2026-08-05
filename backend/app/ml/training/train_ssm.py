"""SSM (State Space Model) training entry point.

This module provides utilities to prepare data loaders and train a
State‑Space Model (SSM) using the same command‑line interface as the
LSTM trainer.  It is intended for offline training runs and produces
model artifacts that can later be loaded by the inference pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ml.features.engineer import engineer_features, create_sequences, add_labels
from app.ml.models.ssm_model import SSMPredictor
from app.ml.training.trainer import train_with_lightning, ARTIFACTS_DIR
from app.utils.logging import logger


def build_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 60,
    batch_size: int = 256,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create training, validation, and test DataLoaders for the SSM.

    The function engineers additional features, adds binary labels based on a
    price‑movement threshold, and then converts the data into fixed‑length
    sequences suitable for time‑series modelling.

    Args:
        df: Raw OHLCV DataFrame indexed by timestamp.
        seq_len: Length of each input sequence (number of timesteps).
        batch_size: Number of samples per batch for the DataLoaders.
        train_frac: Fraction of the dataset to allocate to training.
        val_frac: Fraction of the dataset to allocate to validation.

    Returns:
        A tuple containing the training, validation, and test DataLoaders,
        followed by the number of input features (``n_features``) used by the
        model.
    """
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
    experiment_name: str = "ssm_default",
    d_model: int = 64,
    n_layers: int = 4,
    dropout: float = 0.1,
    seq_len: int = 60,
    max_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Train an SSM model on the provided OHLCV data.

    The function prepares data loaders, instantiates the ``SSMPredictor``,
    runs training via the shared Lightning trainer, and persists the final
    model artifact.

    Args:
        ohlcv_df: DataFrame containing OHLCV columns indexed by timestamp.
        experiment_name: Identifier for the training run; used for artifact paths.
        d_model: Dimensionality of the model’s internal representation.
        n_layers: Number of recurrent layers in the SSM.
        dropout: Dropout probability applied between layers.
        seq_len: Length of the input sequences.
        max_epochs: Maximum number of training epochs.
        batch_size: Batch size for training and validation.
        lr: Learning rate for the optimizer.

    Returns:
        A dictionary containing training metrics and the path to the saved
        model artifact.
    """
    train_loader, val_loader, test_loader, n_features = build_dataloaders(
        ohlcv_df, seq_len=seq_len, batch_size=batch_size
    )

    model = SSMPredictor(
        input_size=n_features,
        d_model=d_model,
        n_layers=n_layers,
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

    save_path = ARTIFACTS_DIR / experiment_name / "final_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "d_model": d_model,
            "n_layers": n_layers,
            "dropout": dropout,
            "seq_len": seq_len,
            "experiment": experiment_name,
        },
        str(save_path),
    )

    results["artifact_path"] = str(save_path)
    logger.info("SSM training complete", **results)
    return results


if __name__ == "__main__":
    """Command‑line interface for training an SSM model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to OHLCV CSV with columns: open,high,low,close,volume",
    )
    parser.add_argument("--name", default="ssm_run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    result = asyncio.run(
        train(
            df,
            experiment_name=args.name,
            max_epochs=args.epochs,
            d_model=args.d_model,
            n_layers=args.n_layers,
        )
    )
    print(json.dumps(result, indent=2))