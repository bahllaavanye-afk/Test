"""
Feature engineering pipeline for ML models.

All features are computed without lookahead bias (shifts are applied where needed).
The resulting feature set is used both for model training and live inference.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.ml.features.advanced_indicators import ADVANCED_FEATURE_COLS, add_advanced_features
from app.ml.features.multi_timeframe import MTF_FEATURE_COLS, add_multi_timeframe_features
from app.ml.features.normalization import FeatureScaler
from app.ml.features.technical import add_technical_features
from app.ml.features.wavelet_features import WAVELET_FEATURE_COLS, add_wavelet_features

logger = logging.getLogger(__name__)

# Social sentiment feature columns added for crypto market_type
SOCIAL_SENTIMENT_FEATURE_COLS: List[str] = [
    "fear_greed_value",
    "fear_greed_7d_avg",
    "fear_greed_change",
    "reddit_mentions",
    "reddit_positive_ratio",
    "reddit_avg_score",
    "is_trending",
    "sentiment_composite",
]

_BASE_FEATURE_COLS: List[str] = [
    # Price-based
    "returns_1",
    "returns_5",
    "returns_10",
    "returns_21",
    # Volatility
    "vol_5",
    "vol_21",
    "vol_63",
    # Trend
    "ema_9_diff",
    "ema_21_diff",
    "ema_50_diff",
    # Momentum
    "rsi_14",
    "rsi_21",
    # MACD
    "macd",
    "macd_signal",
    "macd_hist",
    # Bollinger Bands
    "bb_upper_dist",
    "bb_lower_dist",
    "bb_width",
    # Volume
    "obv_change",
    "volume_ratio",
    # ATR
    "atr_14",
    "atr_pct",
    # Stochastic
    "stoch_k",
    "stoch_d",
    # ADX
    "adx",
]

# Extended feature list: base 27 + advanced + wavelet + multi-timeframe + social sentiment (crypto)
FEATURE_COLS: List[str] = (
    _BASE_FEATURE_COLS
    + ADVANCED_FEATURE_COLS
    + WAVELET_FEATURE_COLS
    + MTF_FEATURE_COLS
    + SOCIAL_SENTIMENT_FEATURE_COLS
)


def engineer_features(
    df: pd.DataFrame,
    normalize: bool = False,
    scaler: Optional[FeatureScaler] = None,
    market_type: str = "equity",
    symbol: str = "",
    social_sentiment: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Apply the full suite of feature engineering steps to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV DataFrame containing at least the columns ``open``, ``high``,
        ``low``, ``close`` and ``volume``.
    normalize : bool, default ``False``
        If ``True``, scale the active feature columns using ``scaler``.
    scaler : FeatureScaler | None, default ``None``
        A pre‑fitted ``FeatureScaler`` instance. Required when ``normalize=True``.
    market_type : str, default ``"equity"``
        Either ``"equity"`` or ``"crypto"``. Crypto markets trigger the addition of
        social‑sentiment features.
    symbol : str, default ``""``
        Ticker symbol (e.g. ``"BTC"``). Used only for crypto sentiment lookup.
    social_sentiment : dict | None, default ``None``
        Pre‑computed sentiment dictionary from
        ``SocialSentimentFeatures.compute_features``. When ``None`` and
        ``market_type=="crypto"``, neutral defaults are used.

    Returns
    -------
    pd.DataFrame
        DataFrame with the engineered feature columns appended. Rows containing
        NaNs from base or advanced indicator calculations are dropped, and any
        remaining NaNs in multi‑timeframe columns are filled with neutral values.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("engineer_features called with non‑DataFrame input: %s", type(df))
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(normalize, bool):
        logger.error("normalize flag must be bool, got %s", type(normalize))
        raise TypeError("normalize must be a boolean")
    if market_type not in {"equity", "crypto"}:
        logger.error("Invalid market_type: %s", market_type)
        raise ValueError("market_type must be either 'equity' or 'crypto'")

    df = df.copy()
    df = add_technical_features(df)
    df = add_advanced_features(df)

    try:
        df = add_wavelet_features(df)
    except Exception as _wv_err:
        logger.error("Wavelet feature generation failed: %s", _wv_err, exc_info=True)

    df = add_multi_timeframe_features(df)

    # ── Social sentiment features (crypto only) ────────────────────────────────
    if market_type == "crypto":
        try:
            from app.ml.features.social_sentiment import SocialSentimentFeatures
        except ImportError as imp_err:
            logger.error("SocialSentimentFeatures module missing: %s", imp_err, exc_info=True)
            raise

        _ssf = SocialSentimentFeatures()
        if social_sentiment is None:
            # Use neutral defaults when no live data is provided (e.g. batch backtest)
            social_sentiment = {}
        row = _ssf.to_dataframe_row(social_sentiment)
        for col in SOCIAL_SENTIMENT_FEATURE_COLS:
            # ``row`` may be an empty Series; ``get`` falls back to a default value.
            df[col] = row.get(col, row[col] if col in row.index else 0.0)

    # Determine which feature columns are actually present (MTF cols depend on TF availability)
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
        try:
            df[active_cols] = scaler.transform(df[active_cols])
        except Exception as norm_err:
            logger.error("Feature scaling failed: %s", norm_err, exc_info=True)
            raise
    elif normalize and scaler is None:
        logger.error("Normalization requested but no scaler provided")
        raise ValueError("Pass a fitted FeatureScaler when normalize=True")

    return df


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 60,
    target_col: str = "target",
) -> Tuple[Union[np.ndarray, Any], Optional[Union[np.ndarray, Any]]]:
    """
    Build sliding‑window sequences for time‑series models (e.g. LSTM, Transformer).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the engineered feature columns and optionally a target column.
    seq_len : int, default ``60``
        Length of each input sequence (number of time steps).
    target_col : str, default ``"target"``
        Name of the column to be used as the regression/classification target.
        If the column is absent, only the feature tensor ``X`` is returned.

    Returns
    -------
    X_out : torch.Tensor | np.ndarray
        Tensor/array of shape ``(n_samples, seq_len, n_features)``.
    y_out : torch.Tensor | np.ndarray | None
        Corresponding targets of shape ``(n_samples,)`` if ``target_col`` exists,
        otherwise ``None``.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("create_sequences received non‑DataFrame input: %s", type(df))
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(seq_len, int) or seq_len <= 0:
        logger.error("Invalid seq_len: %s", seq_len)
        raise ValueError("seq_len must be a positive integer")
    if not isinstance(target_col, str):
        logger.error("target_col must be a string, got %s", type(target_col))
        raise TypeError("target_col must be a string")

    active_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not active_cols:
        logger.error("No engineered feature columns found in DataFrame")
        raise ValueError("DataFrame does not contain any engineered feature columns")

    features = df[active_cols].values
    targets = df[target_col].values if target_col in df.columns else None

    X: List[np.ndarray] = []
    y: List[float] = []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        if targets is not None:
            y.append(targets[i])

    try:
        import torch

        X_out = torch.tensor(np.array(X), dtype=torch.float32)
        y_out = torch.tensor(np.array(y), dtype=torch.float32) if targets is not None else None
    except ImportError as imp_err:
        logger.warning("PyTorch not installed; falling back to NumPy arrays: %s", imp_err)
        X_out = np.array(X, dtype=np.float32)
        y_out = np.array(y, dtype=np.float32) if targets is not None else None

    return X_out, y_out


def add_labels(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.002,
) -> pd.DataFrame:
    """
    Append binary direction labels and a ``target`` alias for training.

    The label is ``1`` when the future return over ``horizon`` bars exceeds ``threshold``,
    otherwise ``0``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a ``close`` price column.
    horizon : int, default ``1``
        Number of bars ahead to compute the forward return.
    threshold : float, default ``0.002``
        Minimum absolute return required to label a move as ``1``.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("add_labels called with non‑DataFrame input: %s", type(df))
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(horizon, int) or horizon <= 0:
        logger.error("Invalid horizon: %s", horizon)
        raise ValueError("horizon must be a positive integer")
    if not isinstance(threshold, (float, int)):
        logger.error("Invalid threshold type: %s", type(threshold))
        raise TypeError("threshold must be a numeric type")

    if "close" not in df.columns:
        logger.error("DataFrame missing required 'close' column for label generation")
        raise KeyError("DataFrame must contain a 'close' column")

    df = df.copy()
    df["future_return"] = df["close"].pct_change(periods=horizon).shift(-horizon)
    df["label"] = (df["future_return"].abs() > threshold).astype(int)
    df["target"] = df["label"]
    df = df.drop(columns=["future_return"])
    return df