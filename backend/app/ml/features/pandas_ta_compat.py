"""
pandas_ta compatibility shim — pure pandas/numpy implementations.

Provides drop-in replacements for the pandas_ta functions used in this codebase.
No external dependencies beyond pandas and numpy.

Implemented:
  rsi, macd, bbands, obv, atr, stoch, adx, cci, ema, supertrend
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# RSI — Wilder's smoothed RSI (EWM with alpha=1/length)
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, length: int = 14) -> pd.Series | None:
    """Relative Strength Index using Wilder's smoothing."""
    if close is None or len(close) < length + 1:
        return None

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    alpha = 1.0 / length
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    result = 100 - (100 / (1 + rs))
    result.name = f"RSI_{length}"
    return result


# ---------------------------------------------------------------------------
# EMA — Exponential Moving Average (helper + standalone)
# ---------------------------------------------------------------------------


def ema(close: pd.Series, length: int = 10) -> pd.Series | None:
    """Exponential Moving Average."""
    if close is None or len(close) < 1:
        return None
    result = close.ewm(span=length, adjust=False).mean()
    result.name = f"EMA_{length}"
    return result


# ---------------------------------------------------------------------------
# MACD — EMA(fast) - EMA(slow), signal = EMA(macd, signal)
# ---------------------------------------------------------------------------


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame | None:
    """MACD line, signal line, and histogram."""
    if close is None or len(close) < slow + signal:
        return None

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    col_macd = f"MACD_{fast}_{slow}_{signal}"
    col_signal = f"MACDs_{fast}_{slow}_{signal}"
    col_hist = f"MACDh_{fast}_{slow}_{signal}"

    return pd.DataFrame(
        {col_macd: macd_line, col_signal: signal_line, col_hist: histogram},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Bollinger Bands — rolling mean ± std * multiplier
# ---------------------------------------------------------------------------


def bbands(
    close: pd.Series,
    length: int = 20,
    std: float = 2.0,
) -> pd.DataFrame | None:
    """Bollinger Bands: upper, lower, middle."""
    if close is None or len(close) < length:
        return None

    mid = close.rolling(window=length).mean()
    rolling_std = close.rolling(window=length).std(ddof=1)
    upper = mid + std * rolling_std
    lower = mid - std * rolling_std

    # pandas_ta uses the numeric std value formatted to 1 decimal place when
    # it's a whole number, e.g. std=2.0 → "2.0".
    std_str = f"{std}"

    col_upper = f"BBU_{length}_{std_str}"
    col_lower = f"BBL_{length}_{std_str}"
    col_mid = f"BBM_{length}_{std_str}"

    return pd.DataFrame(
        {col_upper: upper, col_lower: lower, col_mid: mid},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# OBV — On-Balance Volume
# ---------------------------------------------------------------------------


def obv(close: pd.Series, volume: pd.Series) -> pd.Series | None:
    """On-Balance Volume: cumulative sum of signed volume."""
    if close is None or volume is None or len(close) < 2:
        return None

    direction = np.sign(close.diff()).fillna(0)
    result = (direction * volume).cumsum()
    result.name = "OBV"
    return result


# ---------------------------------------------------------------------------
# ATR — Average True Range (Wilder's EWM smoothing)
# ---------------------------------------------------------------------------


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series | None:
    """Average True Range using Wilder's exponential smoothing."""
    if high is None or low is None or close is None or len(close) < length + 1:
        return None

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / length
    result = tr.ewm(alpha=alpha, adjust=False).mean()
    result.name = f"ATRr_{length}"
    return result


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame | None:
    """
    Stochastic Oscillator.

    %K = (close - lowest_low(k)) / (highest_high(k) - lowest_low(k)) * 100
    Smoothed %K = SMA(%K_raw, smooth_k)   [pandas_ta default smooth_k=3]
    %D = SMA(smoothed_%K, d)

    Column names follow pandas_ta convention:
      STOCHk_{k}_{d}_{smooth_k}
      STOCHd_{k}_{d}_{smooth_k}
    """
    if high is None or low is None or close is None or len(close) < k + d:
        return None

    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()

    stoch_k_raw = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
    stoch_k = stoch_k_raw.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=d).mean()

    col_k = f"STOCHk_{k}_{d}_{smooth_k}"
    col_d = f"STOCHd_{k}_{d}_{smooth_k}"

    return pd.DataFrame({col_k: stoch_k, col_d: stoch_d}, index=close.index)


# ---------------------------------------------------------------------------
# ADX — Average Directional Index (Wilder's smoothing)
# ---------------------------------------------------------------------------


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.DataFrame | None:
    """
    Average Directional Index with +DI and -DI.

    Returns DataFrame with columns: ADX_{length}, DMP_{length}, DMN_{length}
    """
    if high is None or low is None or close is None or len(close) < 2 * length + 1:
        return None

    alpha = 1.0 / length

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder smoothing
    atr_wilder = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr_wilder + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr_wilder + 1e-10)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_val = dx.ewm(alpha=alpha, adjust=False).mean()

    col_adx = f"ADX_{length}"
    col_dmp = f"DMP_{length}"
    col_dmn = f"DMN_{length}"

    return pd.DataFrame(
        {col_adx: adx_val, col_dmp: plus_di, col_dmn: minus_di},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# CCI — Commodity Channel Index
# ---------------------------------------------------------------------------


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    constant: float = 0.015,
) -> pd.Series | None:
    """Commodity Channel Index."""
    if high is None or low is None or close is None or len(close) < length:
        return None

    tp = (high + low + close) / 3.0
    ma = tp.rolling(window=length).mean()
    md = tp.rolling(window=length).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    result = (tp - ma) / (constant * md)
    result.name = f"CCI_{length}"
    return result


# ---------------------------------------------------------------------------
# Supertrend (simplified version)
# ---------------------------------------------------------------------------


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    multiplier: float = 3.0,
) -> pd.Series | None:
    """Simplified Supertrend indicator."""
    if high is None or low is None or close is None or len(close) < length:
        return None

    atr_val = atr(high, low, close, length=length)
    if atr_val is None:
        return None

    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    direction = pd.Series(0, index=close.index)
    direction.iloc[0] = 1

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    result = direction.replace({1: 1, -1: -1})
    result.name = f"SUPER_{length}_{multiplier}"
    return result


# ---------------------------------------------------------------------------
# SIGNAL GENERATION – Enhanced entry/exit logic
# ---------------------------------------------------------------------------


def generate_signal(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    *,
    bb_length: int = 20,
    bb_std: float = 2.0,
    rsi_length: int = 14,
    adx_length: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    adx_max: float = 25.0,
    price_break_percent: float = 0.005,
) -> pd.DataFrame:
    """
    Produce entry and exit signals using a tightened mean‑reversion rule set.

    Entry (long) criteria:
      • Close ≤ lower Bollinger Band * (1 - price_break_percent)
      • RSI < rsi_oversold
      • ADX < adx_max (weak trend – better for mean‑reversion)
      • Volume confirmation – OBV slope positive over the last ``bb_length`` periods

    Exit criteria:
      • Close ≥ middle Bollinger Band (price back to the mean) **or**
      • RSI crosses above rsi_overbought

    The function returns a DataFrame with columns:
      * ``signal`` – 1 for entry, 0 otherwise
      * ``exit``   – 1 for exit, 0 otherwise
      * ``reason`` – textual hint for debugging (optional)
    """
    # Basic validation
    if any(s is None for s in (close, high, low, volume)):
        raise ValueError("All price and volume series must be provided.")
    if len(close) < max(bb_length, rsi_length, adx_length):
        raise ValueError("Insufficient data length for indicator calculations.")

    # Indicator calculations
    bb = bbands(close, length=bb_length, std=bb_std)
    if bb is None:
        raise RuntimeError("Failed to compute Bollinger Bands.")
    rsi_series = rsi(close, length=rsi_length)
    adx_df = adx(high, low, close, length=adx_length)
    obv_series = obv(close, volume)

    # Align all series to the same index (inner join)
    df = pd.concat(
        [close, bb, rsi_series, adx_df, obv_series],
        axis=1,
        join="inner",
    )
    df.columns = [  # ensure deterministic column names
        "close",
        *bb.columns,
        "rsi",
        *adx_df.columns,
        "obv",
    ]

    # Compute OBV slope (simple linear regression over the look‑back window)
    def obv_slope(series: pd.Series) -> pd.Series:
        slope = pd.Series(np.nan, index=series.index)
        window = bb_length
        for i in range(window, len(series)):
            y = series.iloc[i - window : i]
            x = np.arange(window)
            # Least‑squares slope
            if y.isna().any():
                continue
            slope.iloc[i] = np.polyfit(x, y.values, 1)[0]
        return slope

    obv_slope_series = obv_slope(df["obv"])

    # Entry condition
    lower_band = df[bb.columns[0]]
    middle_band = df[bb.columns[2]]
    entry_cond = (
        (df["close"] <= lower_band * (1 - price_break_percent))
        & (df["rsi"] < rsi_oversold)
        & (df[adx_df.columns[0]] < adx_max)
        & (obv_slope_series > 0)
    )

    # Exit condition
    exit_cond = (
        (df["close"] >= middle_band)
        | (df["rsi"] > rsi_overbought)
    )

    signal = pd.Series(0, index=df.index, dtype=int)
    exit_signal = pd.Series(0, index=df.index, dtype=int)

    signal[entry_cond] = 1
    exit_signal[exit_cond] = 1

    # Reason column for diagnostics (optional)
    reason = pd.Series("", index=df.index)
    reason[entry_cond] = "bb_lower+price_break & rsi_oversold & adx_weak & obv_up"
    reason[exit_cond] = "price_mid_cross or rsi_overbought"

    result = pd.DataFrame(
        {
            "signal": signal,
            "exit": exit_signal,
            "reason": reason,
        },
        index=df.index,
    )
    return result


__all__ = [
    "rsi",
    "ema",
    "macd",
    "bbands",
    "obv",
    "atr",
    "stoch",
    "adx",
    "cci",
    "supertrend",
    "generate_signal",
]