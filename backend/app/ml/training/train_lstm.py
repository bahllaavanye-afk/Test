"""LSTM training entry point. Can be run directly or via experiment config."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ml.features.engineer import engineer_features, create_sequences, add_labels
from app.ml.models.lstm import LSTMPredictor
from app.ml.training.trainer import train_with_lightning, ARTIFACTS_DIR
from app.utils.logging import logger


def build_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 60,
    batch_size: int = 256,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build PyTorch DataLoaders for training, validation, and testing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV dataframe.
    seq_len : int
        Length of each input sequence.
    batch_size : int
        Batch size for DataLoaders.
    train_frac : float
        Fraction of data to use for training.
    val_frac : float
        Fraction of data to use for validation.

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, n_features)

    Raises
    ------
    ValueError
        If `train_frac + val_frac` exceeds 1.0, or if `seq_len` is non‑positive,
        or if the resulting dataset is empty.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not (0 <= train_frac <= 1) or not (0 <= val_frac <= 1):
        raise ValueError("train_frac and val_frac must be between 0 and 1")
    if train_frac + val_frac > 1.0:
        raise ValueError("train_frac + val_frac must not exceed 1.0")

    df = engineer_features(df)
    df = add_labels(df, threshold=0.002)
    X, y = create_sequences(df, seq_len=seq_len)

    if len(X) == 0:
        raise ValueError("No sequences generated; check seq_len and input data size")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    n = len(X_t)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
    val_ds = TensorDataset(X_t[n_train : n_train + n_val], y_t[n_train : n_train + n_val])
    test_ds = TensorDataset(X_t[n_train + n_val :], y_t[n_train + n_val :])

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


class TestBuildDataLoaders(unittest.TestCase):
    """Edge‑case tests for `build_dataloaders`."""

    @classmethod
    def setUpClass(cls):
        # Create a minimal realistic DataFrame with required columns
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        data = {
            "open": pd.Series(range(100), index=dates),
            "high": pd.Series(range(100), index=dates),
            "low": pd.Series(range(100), index=dates),
            "close": pd.Series(range(100), index=dates),
            "volume": pd.Series(range(100), index=dates),
        }
        cls.df = pd.DataFrame(data)

    def test_fraction_sum_exceeds_one_raises(self):
        """train_frac + val_frac > 1 should raise ValueError."""
        with self.assertRaises(ValueError):
            build_dataloaders(self.df, train_frac=0.8, val_frac=0.3)

    def test_seq_len_too_large_raises(self):
        """seq_len larger than dataset length should raise ValueError."""
        # Dataset has 100 rows; after feature engineering it will still be 100.
        # With seq_len=200 no sequences can be generated.
        with self.assertRaises(ValueError):
            build_dataloaders(self.df, seq_len=200)

    def test_batch_size_larger_than_dataset(self):
        """When batch_size exceeds dataset size, DataLoader should still produce one batch."""
        # Use a small seq_len to generate many sequences, then set batch_size huge.
        train_loader, val_loader, test_loader, n_features = build_dataloaders(
            self.df, seq_len=10, batch_size=10_000, train_frac=0.6, val_frac=0.2
        )
        # All loaders should have at most one batch
        self.assertLessEqual(len(list(train_loader)), 1)
        self.assertLessEqual(len(list(val_loader)), 1)
        self.assertLessEqual(len(list(test_loader)), 1)
        # n_features should match the engineered feature count (at least 5)
        self.assertGreaterEqual(n_features, 5)


if __name__ == "__main__":
    # Detect if we are running unit tests (environment variable) to avoid
    # interfering with the CLI usage.
    if os.getenv("RUN_UNIT_TESTS") == "1":
        unittest.main(argv=["first-arg-is-ignored"], exit=False)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--csv", required=True, help="Path to OHLCV CSV with columns: open,high,low,close,volume"
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