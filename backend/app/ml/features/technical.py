"""
Technical indicator feature computation using pandas-ta.

All indicators are computed using only past data (no lookahead) to ensure
that the resulting features are suitable for training predictive models
or for live trading. The function operates on a pandas DataFrame that
contains at least a ``close`` price series and optionally ``high``, ``low``
and ``volume`` series. Missing columns default to the ``close`` series (for
``high``/``low``) or a constant series of ones (for ``volume``).

The implementation mirrors the original code base and adds no new
behaviour; it merely enriches the DataFrame with a collection of common
technical features such as returns, volatility, EMA distance, RSI, MACD,
Bollinger Bands, OBV, volume ratio, ATR, Stochastic Oscillator and ADX.

In addition, basic entry/exit signal columns are added.  The signals are
constructed from a combination of the computed indicators and are
intended to provide tighter entry conditions and simple exit filters.
All signal logic uses only information available up to the current bar,
so it is safe for live‑trading pipelines.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import app.ml.features.pandas_ta_compat as ta

_logger = logging.getLogger(__name__)


def _safe_apply(func, *args, **kwargs) -> Optional[pd.DataFrame]:
    """
    Helper to execute a function safely.

    Parameters
    ----------
    func : callable
        The function to execute.
    *args, **kwargs :
        Arguments passed to ``func``.

    Returns
    -------
    Optional[pd.DataFrame]
        The result of ``func`` if successful, otherwise ``None``.
    """
    try:
        return func(*args, **kwargs)
    except (KeyError, ValueError, TypeError) as exc:
        _logger.error(
            "Technical feature computation failed in %s with args=%s, kwargs=%s",
            func.__name__,
            args,
            kwargs,
            exc_info=exc,
        )
        return None


def _get_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """
    Retrieve a column from ``df`` if it exists, otherwise return a series
    filled with ``default`` values.  This utility prevents KeyError when
    optional indicator columns are missing due to earlier failures.
    """
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a suite of technical indicators and append them as new columns
    to a copy of the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input price DataFrame. Must contain a ``close`` column. Optional
        columns are ``high``, ``low`` and ``volume``. Missing optional columns
        are filled with sensible defaults (e.g., ``high`` and ``low`` default
        to ``close``; ``volume`` defaults to a series of ones).

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the original columns plus the technical
        feature columns. If a particular indicator cannot be computed,
        its column is omitted and the error is logged.
    """
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    df = df.copy()
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", pd.Series(1, index=df.index))

    # --- Returns ---
    for n in [1, 5, 10, 21]:
        try:
            df[f"returns_{n}"] = close.pct_change(n)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute returns_%d", n, exc_info=exc)

    # --- Volatility (rolling std of log returns) ---
    try:
        log_ret = np.log(close / close.shift(1))
        for n in [5, 21, 63]:
            df[f"vol_{n}"] = log_ret.rolling(n).std() * np.sqrt(252)
    except Exception as exc:  # pragma: no cover
        _logger.error("Failed to compute volatility features", exc_info=exc)

    # --- EMA distance (normalized) ---
    for span in [9, 21, 50]:
        try:
            ema = close.ewm(span=span).mean()
            df[f"ema_{span}_diff"] = (close - ema) / (ema + 1e-9)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute EMA distance for span %d", span, exc_info=exc)

    # --- RSI ---
    rsi14 = _safe_apply(ta.rsi, close, length=14)
    rsi21 = _safe_apply(ta.rsi, close, length=21)
    if rsi14 is not None:
        df["rsi_14"] = rsi14 / 100.0  # normalize to [0,1]
    if rsi21 is not None:
        df["rsi_21"] = rsi21 / 100.0

    # --- MACD ---
    macd_df = _safe_apply(ta.macd, close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        try:
            df["macd"] = macd_df["MACD_12_26_9"] / (close + 1e-9)
            df["macd_signal"] = macd_df["MACDs_12_26_9"] / (close + 1e-9)
            df["macd_hist"] = macd_df["MACDh_12_26_9"] / (close + 1e-9)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to normalize MACD components", exc_info=exc)

    # --- Bollinger Bands ---
    bb = _safe_apply(ta.bbands, close, length=20, std=2.0)
    if bb is not None:
        try:
            upper = bb["BBU_20_2.0"]
            lower = bb["BBL_20_2.0"]
            mid = bb["BBM_20_2.0"]
            df["bb_upper_dist"] = (upper - close) / (close + 1e-9)
            df["bb_lower_dist"] = (close - lower) / (close + 1e-9)
            df["bb_width"] = (upper - lower) / (mid + 1e-9)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute Bollinger Band features", exc_info=exc)

    # --- OBV change (normalized) ---
    obv = _safe_apply(ta.obv, close, volume)
    if obv is not None:
        try:
            df["obv_change"] = obv.pct_change(5).fillna(0)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute OBV change", exc_info=exc)

    # --- Volume ratio ---
    try:
        vol_ma = volume.rolling(20).mean()
        df["volume_ratio"] = volume / (vol_ma + 1e-9)
    except Exception as exc:  # pragma: no cover
        _logger.error("Failed to compute volume ratio", exc_info=exc)

    # --- ATR ---
    atr = _safe_apply(ta.atr, high, low, close, length=14)
    if atr is not None:
        try:
            df["atr_14"] = atr
            df["atr_pct"] = atr / (close + 1e-9)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute ATR related features", exc_info=exc)

    # --- Stochastic ---
    stoch = _safe_apply(ta.stoch, high, low, close, k=14, d=3)
    if stoch is not None:
        try:
            df["stoch_k"] = stoch["STOCHk_14_3_3"] / 100.0
            df["stoch_d"] = stoch["STOCHd_14_3_3"] / 100.0
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute Stochastic Oscillator features", exc_info=exc)

    # --- ADX ---
    adx_df = _safe_apply(ta.adx, high, low, close, length=14)
    if adx_df is not None:
        try:
            df["adx"] = adx_df["ADX_14"] / 100.0
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute ADX feature", exc_info=exc)

    # ------------------------------------------------------------------
    # Signal generation – tighter entry conditions with confirmation filters
    # ------------------------------------------------------------------
    # Helper series (fallback to zeros/False if a column is missing)
    ema21_diff = _get_series(df, "ema_21_diff")
    rsi14 = _get_series(df, "rsi_14")
    macd_hist = _get_series(df, "macd_hist")
    stoch_k = _get_series(df, "stoch_k")
    stoch_d = _get_series(df, "stoch_d")
    volume_ratio = _get_series(df, "volume_ratio")
    close_shift = close.shift(1)

    # Long entry: price above EMA, low RSI, positive MACD histogram,
    # bullish stochastic crossover, and above‑average volume.
    long_entry = (
        (ema21_diff > 0) &
        (rsi14 < 0.30) &
        (macd_hist > 0) &
        (stoch_k > stoch_d) &
        (volume_ratio > 1.0)
    )
    df["long_entry"] = long_entry.astype(int)

    # Short entry: opposite of long entry with stricter thresholds.
    short_entry = (
        (ema21_diff < 0) &
        (rsi14 > 0.70) &
        (macd_hist < 0) &
        (stoch_k < stoch_d) &
        (volume_ratio < 1.0)
    )
    df["short_entry"] = short_entry.astype(int)

    # Exit logic – triggered when opposite momentum appears or when price
    # moves sharply against the position (1 % move in either direction).
    exit_long = (
        (rsi14 > 0.60) |
        (macd_hist < 0) |
        (close < close_shift * 0.99)
    )
    df["exit_long"] = exit_long.astype(int)

    exit_short = (
        (rsi14 < 0.40) |
        (macd_hist > 0) |
        (close > close_shift * 1.01)
    )
    df["exit_short"] = exit_short.astype(int)

    # Combined signal: 1 = long, -1 = short, 0 = flat.
    # Preference is given to exit signals first.
    combined_signal = (
        -df["exit_short"] +
        df["exit_long"] +
        df["long_entry"] -
        df["short_entry"]
    )
    # Clip to -1, 0, 1
    df["signal"] = combined_signal.clip(-1, 1)

    return df