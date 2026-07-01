"""
Multi-timeframe features: compute indicators on up to 6 timeframes and merge
into the base dataframe. Auto-skips TFs coarser than base or producing < 20 bars.

All features are properly aligned and lagged (shift(1)) to prevent lookahead bias.

Exports:
  add_multi_timeframe_features(df_base, timeframes=None) -> pd.DataFrame
  MTF_FEATURE_COLS: list[str]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from typing import List, Optional

# Canonical ordering of supported timeframes with pandas resample rules
_TF_RULES: dict[str, str] = {
    "5min": "5min",
    "15min": "15min",
    "1h": "1h",
    "4h": "4h",
    "1D": "1D",
    "1W": "1W",
}

# Approximate bar duration in minutes for each TF (used for coarseness check)
_TF_MINUTES: dict[str, float] = {
    "5min": 5,
    "15min": 15,
    "1h": 60,
    "4h": 240,
    "1D": 1440,
    "1W": 10080,
}

ALL_TIMEFRAMES: List[str] = ["5min", "15min", "1h", "4h", "1D", "1W"]


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe using pandas resample."""
    return (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(how="all")
    )


def _detect_base_tf_minutes(df: pd.DataFrame) -> float:
    """
    Estimate the base timeframe in minutes from the median index spacing.
    Falls back to 1 minute if detection fails.
    """
    if len(df) < 2:
        return 1.0
    diffs = pd.Series(df.index).diff().dropna()
    median_td = diffs.median()
    return max(median_td.total_seconds() / 60.0, 1.0)


def _compute_tf_features(tf: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    """
    Compute per‑TF indicator columns on a resampled OHLCV dataframe.
    Returns a DataFrame with columns named tf_{tf_label}_{indicator}.
    """
    out = pd.DataFrame(index=tf.index)
    prefix = f"tf_{tf_label}"
    n = len(tf)

    # RSI(14)
    rsi_len = min(14, n - 1)
    if n > rsi_len:
        rsi_val = ta.rsi(tf["close"], length=rsi_len)
        out[f"{prefix}_rsi"] = rsi_val.fillna(50.0)
    else:
        out[f"{prefix}_rsi"] = pd.Series(50.0, index=tf.index)

    # ADX(14) – provide a sensible default when insufficient data
    adx_len = min(14, n // 3)
    if adx_len >= 3 and n > 2 * adx_len:
        adx_df = ta.adx(tf["high"], tf["low"], tf["close"], length=adx_len)
        col_name = f"ADX_{adx_len}"
        if adx_df is not None and col_name in adx_df.columns:
            out[f"{prefix}_adx"] = adx_df[col_name].fillna(20.0)
        else:
            out[f"{prefix}_adx"] = pd.Series(20.0, index=tf.index)
    else:
        out[f"{prefix}_adx"] = pd.Series(20.0, index=tf.index)

    # Trend: +1 if close > EMA50, -1 otherwise
    ema_len = min(50, max(5, n - 1))
    ema_val = ta.ema(tf["close"], length=ema_len)
    if ema_val is not None:
        out[f"{prefix}_trend"] = np.where(tf["close"] > ema_val, 1.0, -1.0)
    else:
        out[f"{prefix}_trend"] = pd.Series(0.0, index=tf.index)

    # Bollinger Band position: (close - lower) / (upper - lower) clamped [0,1]
    bb_len = min(20, n - 1)
    if bb_len >= 5 and n >= bb_len:
        bb_df = ta.bbands(tf["close"], length=bb_len)
        if bb_df is not None:
            std_str = "2.0"
            upper_col = f"BBU_{bb_len}_{std_str}"
            lower_col = f"BBL_{bb_len}_{std_str}"
            if upper_col in bb_df.columns and lower_col in bb_df.columns:
                band_range = bb_df[upper_col] - bb_df[lower_col]
                bb_pos = (tf["close"] - bb_df[lower_col]) / (band_range + 1e-12)
                out[f"{prefix}_bb_pos"] = bb_pos.clip(0, 1).fillna(0.5)
            else:
                out[f"{prefix}_bb_pos"] = pd.Series(0.5, index=tf.index)
        else:
            out[f"{prefix}_bb_pos"] = pd.Series(0.5, index=tf.index)
    else:
        out[f"{prefix}_bb_pos"] = pd.Series(0.5, index=tf.index)

    # Volume ratio: volume / rolling_mean(volume, 20)
    vol_len = min(20, n - 1)
    if vol_len >= 2:
        avg_vol = tf["volume"].rolling(vol_len, min_periods=2).mean()
        vol_ratio = tf["volume"] / (avg_vol + 1e-12)
        out[f"{prefix}_vol_ratio"] = vol_ratio.clip(0, 10).fillna(1.0)
    else:
        out[f"{prefix}_vol_ratio"] = pd.Series(1.0, index=tf.index)

    # Momentum: pct_change(3) clipped to [-1, 1]
    mom_len = min(3, n - 1)
    if mom_len >= 1:
        momentum = tf["close"].pct_change(mom_len).clip(-1, 1)
        out[f"{prefix}_momentum"] = momentum.fillna(0.0)
    else:
        out[f"{prefix}_momentum"] = pd.Series(0.0, index=tf.index)

    # Garman‑Klass volatility (simplified): rolling mean of 0.5*(ln H/L)^2
    gk = 0.5 * np.log(tf["high"] / tf["low"].replace(0, np.nan)) ** 2
    gk_len = min(10, n - 1)
    if gk_len >= 2:
        gk_mean = gk.rolling(gk_len, min_periods=2).mean()
        gk_vol = gk_mean.apply(lambda x: np.sqrt(max(x, 0)) if not np.isnan(x) else np.nan)
        out[f"{prefix}_gk_vol"] = gk_vol.fillna(0.0)
    else:
        out[f"{prefix}_gk_vol"] = pd.Series(0.0, index=tf.index)

    return out


# Canonical per‑timeframe feature columns produced by _compute_tf_features (prefix
# `tf_{label}`). Imported by app.ml.features.engineer; kept in sync with the function
# above.
_MTF_FEATURE_SUFFIXES = ("rsi", "adx", "trend", "bb_pos", "vol_ratio", "momentum", "gk_vol")
MTF_FEATURE_COLS: list[str] = [
    f"tf_{_tf}_{_suf}" for _tf in _TF_RULES for _suf in _MTF_FEATURE_SUFFIXES
]


def _aggregate_features(df: pd.DataFrame, active_tfs: List[str]) -> pd.DataFrame:
    """
    Compute cross‑timeframe aggregates (mean of each suffix) and a simple
    consensus signal. The aggregates are added to ``df`` in‑place and the
    modified frame is returned.
    """
    if not active_tfs:
        return df

    # Compute mean for each suffix across the active timeframes
    for suffix in _MTF_FEATURE_SUFFIXES:
        cols = [f"tf_{tf}_{suffix}" for tf in active_tfs if f"tf_{tf}_{suffix}" in df.columns]
        if not cols:
            continue
        df[f"mtf_{suffix}_mean"] = df[cols].mean(axis=1)

    # Consensus signal: 1 (long) / -1 (short) / 0 (neutral)
    # Long when:
    #   - RSI mean < 30
    #   - Trend mean > 0
    #   - BB position mean < 0.3 (price near lower band)
    #   - Momentum mean > 0
    #   - ADX mean < 25 (low directional volatility)
    #   - Volume ratio mean > 1.2 (unusual volume)
    long_cond = (
        (df["mtf_rsi_mean"] < 30)
        & (df["mtf_trend_mean"] > 0)
        & (df["mtf_bb_pos_mean"] < 0.3)
        & (df["mtf_momentum_mean"] > 0)
        & (df["mtf_adx_mean"] < 25)
        & (df["mtf_vol_ratio_mean"] > 1.2)
    )

    # Short when the opposite holds
    short_cond = (
        (df["mtf_rsi_mean"] > 70)
        & (df["mtf_trend_mean"] < 0)
        & (df["mtf_bb_pos_mean"] > 0.7)
        & (df["mtf_momentum_mean"] < 0)
        & (df["mtf_adx_mean"] < 25)
        & (df["mtf_vol_ratio_mean"] > 1.2)
    )

    df["mtf_signal"] = 0
    df.loc[long_cond, "mtf_signal"] = 1
    df.loc[short_cond, "mtf_signal"] = -1

    # Exit signal: trigger when any of the long/short conditions reverse
    # (e.g., RSI crosses back toward 50 or trend flips)
    exit_cond = (
        ((df["mtf_signal"] == 1) & (df["mtf_rsi_mean"] >= 45))
        | ((df["mtf_signal"] == -1) & (df["mtf_rsi_mean"] <= 55))
        | (df["mtf_adx_mean"] > 40)  # high ADX suggests strong trend; exit prior position
    )
    df["mtf_exit"] = 0
    df.loc[exit_cond, "mtf_exit"] = 1

    return df


def add_multi_timeframe_features(
    df_base: pd.DataFrame,
    timeframes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Given a base OHLCV dataframe (index=DatetimeIndex), compute features on
    up to 6 timeframes, merge back (forward fill + shift(1)) to prevent lookahead,
    and derive cross‑timeframe aggregate signals.

    Args:
        df_base: OHLCV DataFrame with DatetimeIndex.
        timeframes: List of TF labels from ALL_TIMEFRAMES; default=all 6.

    Returns:
        DataFrame with per‑TF feature columns, aggregated statistics and
        consensus signal columns appended.
    """
    if timeframes is None:
        timeframes = ALL_TIMEFRAMES

    df = df_base.copy()
    df.index = pd.to_datetime(df.index)

    base_min = _detect_base_tf_minutes(df)
    active_tfs: List[str] = []

    for tf_label in timeframes:
        if tf_label not in _TF_RULES:
            continue
        tf_min = _TF_MINUTES[tf_label]

        # Skip TFs equal to or finer than base (no upsampling)
        if tf_min <= base_min * 1.1:
            continue

        rule = _TF_RULES[tf_label]
        try:
            tf = resample_ohlcv(df, rule)
        except Exception:
            continue

        # Skip if fewer than 20 complete bars
        if len(tf) < 20:
            continue

        # Compute indicators on the TF
        tf_feats = _compute_tf_features(tf, tf_label)

        # Merge each feature back: reindex + ffill + shift(1) to avoid lookahead
        for col in tf_feats.columns:
            merged = tf_feats[col].reindex(df.index, method="ffill").shift(1)

            # Neutral fill‑values per feature type
            if col.endswith("_rsi"):
                df[col] = merged.fillna(50.0)
            elif col.endswith("_trend"):
                df[col] = merged.fillna(0.0)
            elif col.endswith("_bb_pos"):
                df[col] = merged.fillna(0.5)
            elif col.endswith("_vol_ratio"):
                df[col] = merged.fillna(1.0)
            elif col.endswith("_momentum"):
                df[col] = merged.fillna(0.0)
            elif col.endswith("_adx"):
                df[col] = merged.fillna(20.0)
            elif col.endswith("_gk_vol"):
                df[col] = merged.fillna(0.0)
            else:
                df[col] = merged.fillna(0.0)

        active_tfs.append(tf_label)

    # Add cross‑timeframe aggregates and consensus signals
    df = _aggregate_features(df, active_tfs)

    return df