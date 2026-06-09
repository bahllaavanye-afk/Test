"""SSM (State Space Model) training entry point. Mirrors train_lstm.py interface."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
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
    experiment_name: str = "ssm_default",
    d_model: int = 64,
    n_layers: int = 4,
    dropout: float = 0.1,
    seq_len: int = 60,
    max_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict:
    train_loader, val_loader, test_loader, n_features = build_dataloaders(ohlcv_df, seq_len, batch_size)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV with columns: open,high,low,close,volume")
    parser.add_argument("--name", default="ssm_run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    result = asyncio.run(
        train(df, experiment_name=args.name, max_epochs=args.epochs, d_model=args.d_model, n_layers=args.n_layers)
    )
    print(json.dumps(result, indent=2))
