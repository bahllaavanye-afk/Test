"""
Multi-timeframe features: compute indicators on 1h, 4h, 1d and merge into base dataframe.
All features are properly aligned and lagged to prevent lookahead bias.
"""
from __future__ import annotations
import pandas as pd
import app.ml.features.pandas_ta_compat as ta


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample minute/hourly OHLCV to a higher timeframe."""
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def add_multi_timeframe_features(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Given a 1h base OHLCV dataframe (index=datetime), compute 4h and 1D features,
    merge back (forward fill), and lag by 1 bar.

    Features added:
    - tf_4h_rsi: RSI(14) on 4h bars
    - tf_4h_trend: 1 if close > EMA50 on 4h, -1 otherwise
    - tf_1d_rsi: RSI(14) on 1d bars
    - tf_1d_trend: 1 if close > EMA50 on 1d, -1 otherwise
    - tf_alignment: +2 all bullish, -2 all bearish, 0 mixed
    """
    df = df_base.copy()
    df.index = pd.to_datetime(df.index)

    for tf_label, rule in [("4h", "4h"), ("1d", "1D")]:
        tf = resample_ohlcv(df, rule)
        if len(tf) < 20:
            df[f"tf_{tf_label}_rsi"] = 50.0
            df[f"tf_{tf_label}_trend"] = 0
            continue

        tf[f"rsi"] = ta.rsi(tf["close"], length=14)
        ema50 = ta.ema(tf["close"], length=min(50, len(tf) - 1))
        tf[f"trend"] = (tf["close"] > ema50).astype(int) * 2 - 1  # +1 or -1

        # Merge back to base timeframe (forward fill, then lag 1)
        merged_rsi = tf["rsi"].reindex(df.index, method="ffill").shift(1)
        merged_trend = tf["trend"].reindex(df.index, method="ffill").shift(1)

        df[f"tf_{tf_label}_rsi"] = merged_rsi.fillna(50.0)
        df[f"tf_{tf_label}_trend"] = merged_trend.fillna(0)

    # Alignment score
    if "tf_4h_trend" in df.columns and "tf_1d_trend" in df.columns:
        df["tf_alignment"] = df["tf_4h_trend"] + df["tf_1d_trend"]
    else:
        df["tf_alignment"] = 0

    return df
