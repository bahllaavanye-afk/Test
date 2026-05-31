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

# Canonical ordering of supported timeframes with pandas resample rules
_TF_RULES: dict[str, str] = {
    "5min":  "5min",
    "15min": "15min",
    "1h":    "1h",
    "4h":    "4h",
    "1D":    "1D",
    "1W":    "1W",
}

# Approximate bar duration in minutes for each TF (used for coarseness check)
_TF_MINUTES: dict[str, float] = {
    "5min":  5,
    "15min": 15,
    "1h":    60,
    "4h":    240,
    "1D":    1440,
    "1W":    10080,
}

ALL_TIMEFRAMES = ["5min", "15min", "1h", "4h", "1D", "1W"]


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe using pandas resample."""
    return df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(how="all")


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
    Compute per-TF indicator columns on a resampled OHLCV dataframe.
    Returns a DataFrame with columns named tf_{tf_label}_{indicator}.
    """
    out = pd.DataFrame(index=tf.index)
    prefix = f"tf_{tf_label}"
    n = len(tf)

    # RSI(14)
    rsi_len = min(14, n - 1)
    rsi_val = ta.rsi(tf["close"], length=rsi_len) if n > rsi_len else None
    out[f"{prefix}_rsi"] = rsi_val if rsi_val is not None else pd.Series(50.0, index=tf.index)

    # ADX(14)
    adx_len = min(14, n // 3)
    if adx_len >= 3 and n > 2 * adx_len:
        adx_df = ta.adx(tf["high"], tf["low"], tf["close"], length=adx_len)
        out[f"{prefix}_adx"] = adx_df[f"ADX_{adx_len}"] if adx_df is not None else 20.0
    else:
        out[f"{prefix}_adx"] = 20.0

    # Trend: +1 if close > EMA50, -1 otherwise
    ema_len = min(50, max(5, n - 1))
    ema_val = ta.ema(tf["close"], length=ema_len)
    if ema_val is not None:
        out[f"{prefix}_trend"] = np.where(tf["close"] > ema_val, 1.0, -1.0)
    else:
        out[f"{prefix}_trend"] = 0.0

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
                out[f"{prefix}_bb_pos"] = bb_pos.clip(0, 1)
            else:
                out[f"{prefix}_bb_pos"] = 0.5
        else:
            out[f"{prefix}_bb_pos"] = 0.5
    else:
        out[f"{prefix}_bb_pos"] = 0.5

    # Volume ratio: volume / rolling_mean(volume, 20)
    vol_len = min(20, n - 1)
    if vol_len >= 2:
        avg_vol = tf["volume"].rolling(vol_len, min_periods=2).mean()
        out[f"{prefix}_vol_ratio"] = (tf["volume"] / (avg_vol + 1e-12)).clip(0, 10)
    else:
        out[f"{prefix}_vol_ratio"] = 1.0

    # Momentum: pct_change(3) clipped to [-1, 1]
    mom_len = min(3, n - 1)
    if mom_len >= 1:
        out[f"{prefix}_momentum"] = tf["close"].pct_change(mom_len).clip(-1, 1)
    else:
        out[f"{prefix}_momentum"] = 0.0

    # Garman-Klass vol (simplified): rolling mean of 0.5*(ln H/L)^2
    gk = (0.5 * np.log(tf["high"] / tf["low"].replace(0, np.nan)) ** 2)
    gk_len = min(10, n - 1)
    if gk_len >= 2:
        gk_mean = gk.rolling(gk_len, min_periods=2).mean()
        out[f"{prefix}_gk_vol"] = gk_mean.apply(lambda x: np.sqrt(max(x, 0)) if not np.isnan(x) else np.nan)
    else:
        out[f"{prefix}_gk_vol"] = 0.0

    return out


def add_multi_timeframe_features(
    df_base: pd.DataFrame,
    timeframes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Given a base OHLCV dataframe (index=DatetimeIndex), compute features on
    up to 6 timeframes, merge back (forward fill + shift(1)) to prevent lookahead.

    Args:
        df_base: OHLCV DataFrame with DatetimeIndex
        timeframes: list of TF labels from ALL_TIMEFRAMES; default=all 6

    Returns:
        df with per-TF feature columns + cross-TF aggregate columns appended.
    """
    if timeframes is None:
        timeframes = ALL_TIMEFRAMES

    df = df_base.copy()
    df.index = pd.to_datetime(df.index)

    base_min = _detect_base_tf_minutes(df)
    active_tfs: list[str] = []

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
            # Fill NaNs with neutral values based on feature type
            if "_rsi" in col:
                df[col] = merged.fillna(50.0)
            elif "_trend" in col:
                df[col] = merged.fillna(0.0)
            elif "_bb_pos" in col:
                df[col] = merged.fillna(0.5)
            elif "_vol_ratio" in col:
                df[col] = merged.fillna(1.0)
            else:
                df[col] = merged.fillna(0.0)

        active_tfs.append(tf_label)

    # -----------------------------------------------------------------------
    # Cross-TF aggregate features
    # -----------------------------------------------------------------------
    trend_cols = [f"tf_{tf}_trend" for tf in active_tfs if f"tf_{tf}_trend" in df.columns]
    momentum_cols = [f"tf_{tf}_momentum" for tf in active_tfs if f"tf_{tf}_momentum" in df.columns]
    vol_cols = [f"tf_{tf}_vol_ratio" for tf in active_tfs if f"tf_{tf}_vol_ratio" in df.columns]

    if trend_cols:
        trend_mat = df[trend_cols]
        df["tf_trend_score"] = trend_mat.sum(axis=1)
        df["tf_bull_count"] = (trend_mat > 0).sum(axis=1).astype(float)
        df["tf_bear_count"] = (trend_mat < 0).sum(axis=1).astype(float)
        # Divergence: 1 if any TF disagrees with majority trend
        majority = np.sign(df["tf_trend_score"])
        df["tf_trend_divergence"] = (
            trend_mat.apply(lambda row: int(any(row * majority[row.name] < 0)), axis=1)
        ).astype(float)
    else:
        df["tf_trend_score"] = 0.0
        df["tf_bull_count"] = 0.0
        df["tf_bear_count"] = 0.0
        df["tf_trend_divergence"] = 0.0

    if momentum_cols:
        mom_mat = df[momentum_cols]
        n_tfs = len(momentum_cols)
        df["tf_momentum_score"] = mom_mat.sum(axis=1) / max(n_tfs, 1)
    else:
        df["tf_momentum_score"] = 0.0

    if vol_cols:
        vol_mat = df[vol_cols]
        df["tf_vol_agreement"] = vol_mat.std(axis=1).fillna(0.0)
    else:
        df["tf_vol_agreement"] = 0.0

    return df


def _build_mtf_feature_cols(timeframes: list[str] | None = None) -> list[str]:
    """Build the expected MTF feature column list for the given TF set."""
    if timeframes is None:
        timeframes = ALL_TIMEFRAMES
    per_tf_suffixes = [
        "_rsi", "_adx", "_trend", "_bb_pos", "_vol_ratio", "_momentum", "_gk_vol"
    ]
    cols: list[str] = []
    for tf_label in timeframes:
        for suf in per_tf_suffixes:
            cols.append(f"tf_{tf_label}{suf}")
    # Cross-TF aggregates
    cols += [
        "tf_trend_score",
        "tf_momentum_score",
        "tf_bull_count",
        "tf_bear_count",
        "tf_vol_agreement",
        "tf_trend_divergence",
    ]
    return cols


# Exported column list (assumes all 6 TFs are active; engineer.py should use
# only columns actually present in the DataFrame after the call)
MTF_FEATURE_COLS: list[str] = _build_mtf_feature_cols(ALL_TIMEFRAMES)
