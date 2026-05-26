"""
Master feature engineering pipeline.
All features are computed without lookahead bias (shift applied where needed).
Features are used both for ML training and live inference.
"""
import pandas as pd
import numpy as np
from app.ml.features.technical import add_technical_features
from app.ml.features.normalization import FeatureScaler


FEATURE_COLS = [
    # Price-based
    "returns_1", "returns_5", "returns_10", "returns_21",
    # Volatility
    "vol_5", "vol_21", "vol_63",
    # Trend
    "ema_9_diff", "ema_21_diff", "ema_50_diff",
    # Momentum
    "rsi_14", "rsi_21",
    # MACD
    "macd", "macd_signal", "macd_hist",
    # Bollinger Bands
    "bb_upper_dist", "bb_lower_dist", "bb_width",
    # Volume
    "obv_change", "volume_ratio",
    # ATR
    "atr_14", "atr_pct",
    # Stochastic
    "stoch_k", "stoch_d",
    # ADX
    "adx",
]


def engineer_features(df: pd.DataFrame, normalize: bool = False, scaler: FeatureScaler | None = None) -> pd.DataFrame:
    """
    Apply all feature engineering to an OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        normalize: if True, apply StandardScaler
        scaler: pre-fitted FeatureScaler for inference

    Returns:
        DataFrame with added feature columns
    """
    df = df.copy()
    df = add_technical_features(df)

    # Drop rows with NaN from indicator computation
    df = df.dropna(subset=FEATURE_COLS)

    if normalize and scaler is not None:
        df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    elif normalize and scaler is None:
        raise ValueError("Pass a fitted FeatureScaler when normalize=True")

    return df


def create_sequences(df: pd.DataFrame, seq_len: int = 60, target_col: str = "target") -> tuple:
    """
    Create (X, y) for LSTM training.
    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples,)
    """
    features = df[FEATURE_COLS].values
    targets = df[target_col].values if target_col in df.columns else None

    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        if targets is not None:
            y.append(targets[i])

    import torch
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32) if targets is not None else None
    return X_tensor, y_tensor


def add_labels(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.002) -> pd.DataFrame:
    """
    Add binary direction label.
    1 = price goes up by > threshold over horizon bars
    0 = price goes down or stays flat
    """
    df = df.copy()
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    df["label"] = (future_return > threshold).astype(int)
    df["target"] = df["label"]  # alias for create_sequences compatibility
    return df.dropna(subset=["label"])
