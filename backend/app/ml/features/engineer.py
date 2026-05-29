"""
Master feature engineering pipeline.
All features are computed without lookahead bias (shift applied where needed).
Features are used both for ML training and live inference.
"""
import pandas as pd
import numpy as np
from app.ml.features.technical import add_technical_features
from app.ml.features.advanced_indicators import add_advanced_features, ADVANCED_FEATURE_COLS
from app.ml.features.wavelet_features import add_wavelet_features, WAVELET_FEATURE_COLS
from app.ml.features.multi_timeframe import add_multi_timeframe_features, MTF_FEATURE_COLS
from app.ml.features.normalization import FeatureScaler


_BASE_FEATURE_COLS = [
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

# Extended feature list: base 27 + advanced + wavelet + multi-timeframe
FEATURE_COLS = _BASE_FEATURE_COLS + ADVANCED_FEATURE_COLS + WAVELET_FEATURE_COLS + MTF_FEATURE_COLS


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
    df = add_advanced_features(df)
    try:
        df = add_wavelet_features(df)
    except Exception as _wv_err:
        print(f"[engineer] wavelet features skipped: {_wv_err}", flush=True)
    df = add_multi_timeframe_features(df)

    # Determine which feature cols are actually present (MTF cols depend on TF availability)
    active_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Drop rows with NaN from indicator computation (only on base + advanced cols)
    base_and_adv = [c for c in _BASE_FEATURE_COLS + ADVANCED_FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=base_and_adv)

    # Fill any remaining NaNs in MTF cols with neutral values
    mtf_cols_present = [c for c in MTF_FEATURE_COLS if c in df.columns]
    for col in mtf_cols_present:
        if "_rsi" in col:
            df[col] = df[col].fillna(50.0)
        elif "_trend" in col or "regime" in col:
            df[col] = df[col].fillna(0.0)
        elif "_bb_pos" in col:
            df[col] = df[col].fillna(0.5)
        elif "_vol_ratio" in col:
            df[col] = df[col].fillna(1.0)
        else:
            df[col] = df[col].fillna(0.0)

    if normalize and scaler is not None:
        df[active_cols] = scaler.transform(df[active_cols])
    elif normalize and scaler is None:
        raise ValueError("Pass a fitted FeatureScaler when normalize=True")

    return df


def create_sequences(df: pd.DataFrame, seq_len: int = 60, target_col: str = "target") -> tuple:
    """
    Create (X, y) for LSTM/Transformer training.
    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples,)

    Uses only feature columns actually present in df to handle partial TF availability.
    """
    active_cols = [c for c in FEATURE_COLS if c in df.columns]
    features = df[active_cols].values
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
