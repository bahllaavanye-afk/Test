"""
pandas_ta compatibility shim — pure pandas/numpy implementations.

Provides drop‑in replacements for the pandas_ta functions used in this codebase.
No external dependencies beyond pandas and numpy.

Implemented:
  rsi, macd, bbands, obv, atr, stoch, adx, cci, ema, supertrend
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "rsi",
    "ema",
    "macd",
    "bbands",
    "obv",
    "atr",
    "stoch",
    "adx",
    # CCI and supertrend are defined later in the file (not shown here)
]


def rsi(close: pd.Series, length: int = 14) -> Optional[pd.Series]:
    """
    Calculate Wilder's Relative Strength Index (RSI).

    Args:
        close: Series of closing prices.
        length: Look‑back period for the RSI calculation (default ``14``).

    Returns:
        A pandas Series containing the RSI values, named ``RSI_<length>``,
        or ``None`` if the input series is ``None`` or too short.
    """
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


def ema(close: pd.Series, length: int = 10) -> Optional[pd.Series]:
    """
    Compute the Exponential Moving Average (EMA).

    Args:
        close: Series of closing prices.
        length: EMA span (default ``10``).

    Returns:
        A pandas Series with the EMA values, named ``EMA_<length>``,
        or ``None`` if the input series is ``None`` or empty.
    """
    if close is None or len(close) < 1:
        return None
    result = close.ewm(span=length, adjust=False).mean()
    result.name = f"EMA_{length}"
    return result


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[pd.DataFrame]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

    The MACD line is the difference between a fast and a slow EMA.
    The signal line is an EMA of the MACD line, and the histogram is the
    difference between the MACD line and the signal line.

    Args:
        close: Series of closing prices.
        fast: Span for the fast EMA (default ``12``).
        slow: Span for the slow EMA (default ``26``).
        signal: Span for the signal EMA (default ``9``).

    Returns:
        A DataFrame with columns ``MACD_<fast>_<slow>_<signal>``,
        ``MACDs_<fast>_<slow>_<signal>``, and ``MACDh_<fast>_<slow>_<signal>``,
        indexed like ``close``, or ``None`` if the input series is ``None`` or
        insufficiently long.
    """
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


def bbands(
    close: pd.Series,
    length: int = 20,
    std: float = 2.0,
) -> Optional[pd.DataFrame]:
    """
    Compute Bollinger Bands.

    The middle band is a simple moving average; the upper and lower bands are
    the middle band plus/minus ``std`` times the rolling standard deviation.

    Args:
        close: Series of closing prices.
        length: Look‑back period for the moving average (default ``20``).
        std: Number of standard deviations for the band width (default ``2.0``).

    Returns:
        A DataFrame with columns ``BBU_<length>_<std>``, ``BBL_<length>_<std>``,
        and ``BBM_<length>_<std>``, indexed like ``close``, or ``None`` if the
        input series is ``None`` or too short.
    """
    if close is None or len(close) < length:
        return None

    mid = close.rolling(window=length).mean()
    rolling_std = close.rolling(window=length).std(ddof=1)
    upper = mid + std * rolling_std
    lower = mid - std * rolling_std

    # pandas_ta formats the std value as a string, preserving any trailing
    # zero (e.g. 2.0 → "2.0").
    std_str = f"{std}"

    col_upper = f"BBU_{length}_{std_str}"
    col_lower = f"BBL_{length}_{std_str}"
    col_mid = f"BBM_{length}_{std_str}"

    return pd.DataFrame(
        {col_upper: upper, col_lower: lower, col_mid: mid},
        index=close.index,
    )


def obv(close: pd.Series, volume: pd.Series) -> Optional[pd.Series]:
    """
    Calculate On‑Balance Volume (OBV).

    OBV is a cumulative sum of volume, weighted by the direction of price
    changes.

    Args:
        close: Series of closing prices.
        volume: Series of traded volume.

    Returns:
        A pandas Series named ``OBV`` containing the cumulative OBV values,
        or ``None`` if inputs are ``None`` or insufficiently long.
    """
    if close is None or volume is None or len(close) < 2:
        return None

    direction = np.sign(close.diff()).fillna(0)
    result = (direction * volume).cumsum()
    result.name = "OBV"
    return result


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> Optional[pd.Series]:
    """
    Compute the Average True Range (ATR) using Wilder's smoothing.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        length: Look‑back period for the ATR (default ``14``).

    Returns:
        A pandas Series named ``ATRr_<length>`` with the ATR values,
        or ``None`` if any input series is ``None`` or too short.
    """
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


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Stochastic Oscillator.

    The %K line measures the position of the close relative to the recent
    high‑low range, optionally smoothed. The %D line is a moving average of %K.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        k: Look‑back period for the high/low range (default ``14``).
        d: Period for the %D moving average (default ``3``).
        smooth_k: Period for smoothing %K (default ``3``).

    Returns:
        A DataFrame with columns ``STOCHk_<k>_<d>_<smooth_k>`` and
        ``STOCHd_<k>_<d>_<smooth_k>``, indexed like ``close``, or ``None`` if the
        inputs are ``None`` or insufficiently long.
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


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> Optional[pd.DataFrame]:
    """
    Average Directional Index (ADX) with +DI and -DI components.

    ADX measures trend strength, while +DI and -DI indicate the direction of
    the trend.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        length: Smoothing period (default ``14``).

    Returns:
        A DataFrame with columns ``ADX_<length>``, ``DMP_<length>`` (plus DI),
        and ``DMN_<length>`` (minus DI), indexed like ``close``, or ``None`` if
        inputs are ``None`` or too short.
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
# (Implementation continues below in the original file)