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
# --------------------------------
# ... (truncated for brevity)

# ---------------------------------------------------------------------------
# Unit Tests for Edge Cases
# ---------------------------------------------------------------------------

import unittest
import pandas.testing as pdt


class TestPandasTaCompatEdgeCases(unittest.TestCase):
    """Edge‑case tests for pandas_ta compatibility shim."""

    def test_rsi_insufficient_length_returns_none(self):
        """RSI should return None when the series is shorter than length+1."""
        close = pd.Series([100, 101, 102])  # length 3 < 14+1
        result = rsi(close, length=14)
        self.assertIsNone(result)

    def test_bbands_column_naming_with_integer_std(self):
        """BBands column names must reflect the raw std value (int vs float)."""
        # Generate a series long enough for default length=20
        close = pd.Series(np.arange(30, dtype=float))
        df = bbands(close, length=20, std=2)  # std passed as int
        expected_upper = "BBU_20_2"
        expected_lower = "BBL_20_2"
        expected_mid = "BBM_20_2"
        self.assertIn(expected_upper, df.columns)
        self.assertIn(expected_lower, df.columns)
        self.assertIn(expected_mid, df.columns)

    def test_stoch_zero_k_raises(self):
        """Providing k=0 to stoch should raise a ValueError from pandas."""
        high = pd.Series([10, 12, 14, 16, 18])
        low = pd.Series([5, 6, 7, 8, 9])
        close = pd.Series([7, 9, 11, 13, 15])
        # k=0 leads to pandas rolling with window=0 which raises ValueError
        with self.assertRaises(ValueError):
            stoch(high, low, close, k=0, d=3, smooth_k=3)

    def test_ema_none_input_returns_none(self):
        """EMA should return None when input series is None."""
        self.assertIsNone(ema(None, length=10))


if __name__ == "__main__":
    unittest.main()