"""
Master feature engineering pipeline.
All features are computed without lookahead bias (shift applied where needed).
Features are used both for ML training and live inference.
"""
import numpy as np
import pandas as pd

from app.ml.features.advanced_indicators import ADVANCED_FEATURE_COLS, add_advanced_features
from app.ml.features.alternative import ALTERNATIVE_FEATURE_COLS, add_alternative_features
from app.ml.features.microstructure import MICROSTRUCTURE_FEATURE_COLS, add_vpin_feature
from app.ml.features.multi_timeframe import MTF_FEATURE_COLS, add_multi_timeframe_features
from app.ml.features.normalization import FeatureScaler
from app.ml.features.technical import add_technical_features
from app.ml.features.wavelet_features import WAVELET_FEATURE_COLS, add_wavelet_features

# Social sentiment feature columns added for crypto market_type
SOCIAL_SENTIMENT_FEATURE_COLS = [
    "fear_greed_value",
    "fear_greed_7d_avg",
    "fear_greed_change",
    "reddit_mentions",
    "reddit_positive_ratio",
    "reddit_avg_score",
    "is_trending",
    "sentiment_composite",
]

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

# Extended feature list: base 27 + advanced + wavelet + multi-timeframe + social sentiment (crypto)
# + microstructure (VPIN, LOB imbalance, spread) + alternative data (funding rates, OI) for crypto
FEATURE_COLS = (
    _BASE_FEATURE_COLS
    + ADVANCED_FEATURE_COLS
    + WAVELET_FEATURE_COLS
    + MTF_FEATURE_COLS
    + SOCIAL_SENTIMENT_FEATURE_COLS
    + MICROSTRUCTURE_FEATURE_COLS  # lob_imbalance, spread_bps
    + ["vpin"]                     # VPIN rolling (VPINFeatures)
    + ALTERNATIVE_FEATURE_COLS     # funding_rate, funding_rate_ma7, oi_change_pct, oi_momentum
)

def engineer_features(
    df: pd.DataFrame,
    normalize: bool = False,
    scaler: FeatureScaler | None = None,
    market_type: str = "equity",
    symbol: str = "",
    social_sentiment: dict | None = None,
) -> pd.DataFrame:
    """
    Apply all feature engineering to an OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        normalize: if True, apply StandardScaler
        scaler: pre-fitted FeatureScaler for inference
        market_type: "crypto" or "equity" — enables crypto-specific features
        symbol: ticker symbol (e.g. "BTC") used for crypto sentiment lookup
        social_sentiment: pre-computed dict from SocialSentimentFeatures.compute_features().
            If None and market_type=="crypto", features are left at neutral defaults.
            Pass a pre-awaited result to avoid blocking the sync pipeline.

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

    # ── Social sentiment features (crypto only) ────────────────────────────────
    # For crypto symbols, enrich the feature matrix with social/sentiment signals.
    # Callers in async contexts should pre-compute features via:
    #   from app.ml.features.social_sentiment import SocialSentimentFeatures
    #   sentiment_feats = await SocialSentimentFeatures().compute_features(symbol)
    # and pass the result as social_sentiment=sentiment_feats.
    if market_type == "crypto":
        from app.ml.features.social_sentiment import SocialSentimentFeatures
        _ssf = SocialSentimentFeatures()
        if social_sentiment is None:
            # Use neutral defaults when no live data is provided (e.g. batch backtest)
            social_sentiment = {}
        row = _ssf.to_dataframe_row(social_sentiment)
        for col in SOCIAL_SENTIMENT_FEATURE_COLS:
            df[col] = row.get(col, row[col] if col in row.index else 0.0)

    # ── VPIN (Volume-Synchronized Probability of Informed Trading) ─────────────
    # Requires 'volume' column. NaN for first 49 rows (warmup).
    if "volume" in df.columns:
        try:
            df = add_vpin_feature(df, window=50)
        except Exception as _vpin_err:
            print(f"[engineer] VPIN skipped: {_vpin_err}", flush=True)
            df["vpin"] = 0.0
    else:
        df["vpin"] = 0.0
    df["vpin"] = df["vpin"].fillna(0.0)

    # ── Binance funding rates + open interest (crypto only) ───────────────────
    # Uses public Binance Futures REST — no auth needed. Returns NaN for equities.
    if market_type == "crypto" and symbol:
        try:
            df = add_alternative_features(df, symbol=symbol)
        except Exception as _alt_err:
            print(f"[engineer] alternative features skipped: {_alt_err}", flush=True)
            for col in ALTERNATIVE_FEATURE_COLS:
                if col not in df.columns:
                    df[col] = np.nan
    else:
        for col in ALTERNATIVE_FEATURE_COLS:
            if col not in df.columns:
                df[col] = np.nan
    for col in ALTERNATIVE_FEATURE_COLS:
        df[col] = df[col].fillna(0.0)

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

    try:
        import torch
        X_out = torch.tensor(np.array(X), dtype=torch.float32)
        y_out = torch.tensor(np.array(y), dtype=torch.float32) if targets is not None else None
    except ImportError:
        # torch not installed (CI / Render free tier) — return numpy arrays.
        # Both torch tensors and numpy arrays expose .shape, so downstream
        # shape checks and array-accepting loaders still work.
        X_out = np.array(X, dtype=np.float32)
        y_out = np.array(y, dtype=np.float32) if targets is not None else None
    return X_out, y_out


def add_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    entry_threshold: float = 0.005,
    exit_threshold: float = -0.005,
    confirmation_window: int = 3,
) -> pd.DataFrame:
    """
    Generate target labels for supervised learning.

    Labels are based on forward‑looking returns while respecting
    a no‑lookahead policy (future returns are shifted).

    * Long entry (label 1) – forward return > ``entry_threshold`` **and**
      confirmation filters:
        - recent short‑term returns are positive,
        - RSI is in a moderate range (40‑60),
        - MACD histogram is positive,
        - ATR% is below 2 % (low volatility).

    * Short entry (label -1) – forward return < ``exit_threshold``.

    * Otherwise label 0 (no clear signal).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``close`` and the technical columns used
        for confirmation filters.
    horizon : int, default 5
        Number of periods ahead to compute the forward return.
    entry_threshold : float, default 0.005
        Minimum forward return (5 bps) to consider a long signal.
    exit_threshold : float, default -0.005
        Maximum forward return (‑5 bps) to consider a short signal.
    confirmation_window : int, default 3
        Number of recent return columns to average for additional confirmation.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``target`` column.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1) Compute forward return without lookahead bias.
    # ------------------------------------------------------------------
    df["future_ret"] = df["close"].shift(-horizon) / df["close"] - 1.0

    # ------------------------------------------------------------------
    # 2) Base entry/exit conditions.
    # ------------------------------------------------------------------
    long_cond = df["future_ret"] > entry_threshold
    short_cond = df["future_ret"] < exit_threshold

    # ------------------------------------------------------------------
    # 3) Confirmation filters for long entries.
    # ------------------------------------------------------------------
    # Recent returns (e.g., returns_1, returns_5, returns_10) – at least one
    # must be positive to avoid entering on a single‑period spike.
    return_cols = [col for col in df.columns if col.startswith("returns_")]
    if return_cols:
        recent_positive = df[return_cols].iloc[:, -confirmation_window:].gt(0).any(axis=1)
        long_cond &= recent_positive

    # RSI moderation
    if "rsi_14" in df.columns:
        long_cond &= df["rsi_14"].between(40, 60)

    # MACD histogram positivity
    if "macd_hist" in df.columns:
        long_cond &= df["macd_hist"] > 0

    # Low volatility filter (ATR% < 2%)
    if "atr_pct" in df.columns:
        long_cond &= df["atr_pct"] < 0.02

    # ------------------------------------------------------------------
    # 4) Assemble final label column.
    # ------------------------------------------------------------------
    df["target"] = 0
    df.loc[long_cond, "target"] = 1
    df.loc[short_cond, "target"] = -1

    # ------------------------------------------------------------------
    # 5) Clean‑up: drop helper column and any rows that cannot be labeled
    #    because the forward horizon extends beyond the data.
    # ------------------------------------------------------------------
    df.drop(columns=["future_ret"], inplace=True)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    return df