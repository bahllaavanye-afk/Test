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
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import app.ml.features.pandas_ta_compat as ta

_logger = logging.getLogger(__name__)

# Column name constants
CLOSE_COL = "close"
HIGH_COL = "high"
LOW_COL = "low"
VOLUME_COL = "volume"

# Numerical constants
EPS = 1e-9

# Returns periods
RETURN_PERIODS = [1, 5, 10, 21]

# Volatility rolling windows
VOLATILITY_WINDOWS = [5, 21, 63]

# EMA spans
EMA_SPANS = [9, 21, 50]

# RSI lengths
RSI_LENGTHS = [14, 21]

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_COL = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
MACD_SIGNAL_COL = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
MACD_HIST_COL = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"

# Bollinger Bands parameters
BB_LENGTH = 20
BB_STD = 2.0
BB_UPPER_COL = f"BBU_{BB_LENGTH}_{BB_STD}"
BB_LOWER_COL = f"BBL_{BB_LENGTH}_{BB_STD}"
BB_MID_COL = f"BBM_{BB_LENGTH}_{BB_STD}"

# OBV change period
OBV_PCT_CHANGE = 5

# Volume ratio rolling window
VOLUME_RATIO_WINDOW = 20

# ATR length
ATR_LENGTH = 14

# Stochastic parameters
STOCH_K = 14
STOCH_D = 3
STOCH_K_COL = f"STOCHk_{STOCH_K}_{STOCH_D}_{STOCH_D}"
STOCH_D_COL = f"STOCHd_{STOCH_K}_{STOCH_D}_{STOCH_D}"

# ADX length
ADX_LENGTH = 14
ADX_COL = f"ADX_{ADX_LENGTH}"


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
    if CLOSE_COL not in df.columns:
        raise ValueError(f"Input DataFrame must contain a '{CLOSE_COL}' column.")

    df = df.copy()
    close = df[CLOSE_COL]
    high = df.get(HIGH_COL, close)
    low = df.get(LOW_COL, close)
    volume = df.get(VOLUME_COL, pd.Series(1, index=df.index))

    # --- Returns ---
    for n in RETURN_PERIODS:
        try:
            df[f"returns_{n}"] = close.pct_change(n)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute returns_%d", n, exc_info=exc)

    # --- Volatility (rolling std of log returns) ---
    try:
        log_ret = np.log(close / close.shift(1))
        for n in VOLATILITY_WINDOWS:
            df[f"vol_{n}"] = log_ret.rolling(n).std() * np.sqrt(252)
    except Exception as exc:  # pragma: no cover
        _logger.error("Failed to compute volatility features", exc_info=exc)

    # --- EMA distance (normalized) ---
    for span in EMA_SPANS:
        try:
            ema = close.ewm(span=span).mean()
            df[f"ema_{span}_diff"] = (close - ema) / (ema + EPS)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute EMA distance for span %d", span, exc_info=exc)

    # --- RSI ---
    for length in RSI_LENGTHS:
        rsi = _safe_apply(ta.rsi, close, length=length)
        if rsi is not None:
            df[f"rsi_{length}"] = rsi / 100.0  # normalize to [0,1]

    # --- MACD ---
    macd_df = _safe_apply(ta.macd, close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd_df is not None:
        try:
            df["macd"] = macd_df[MACD_COL] / (close + EPS)
            df["macd_signal"] = macd_df[MACD_SIGNAL_COL] / (close + EPS)
            df["macd_hist"] = macd_df[MACD_HIST_COL] / (close + EPS)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to normalize MACD components", exc_info=exc)

    # --- Bollinger Bands ---
    bb = _safe_apply(ta.bbands, close, length=BB_LENGTH, std=BB_STD)
    if bb is not None:
        try:
            upper = bb[BB_UPPER_COL]
            lower = bb[BB_LOWER_COL]
            mid = bb[BB_MID_COL]
            df["bb_upper_dist"] = (upper - close) / (close + EPS)
            df["bb_lower_dist"] = (close - lower) / (close + EPS)
            df["bb_width"] = (upper - lower) / (mid + EPS)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute Bollinger Band features", exc_info=exc)

    # --- OBV change (normalized) ---
    obv = _safe_apply(ta.obv, close, volume)
    if obv is not None:
        try:
            df["obv_change"] = obv.pct_change(OBV_PCT_CHANGE).fillna(0)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute OBV change", exc_info=exc)

    # --- Volume ratio ---
    try:
        vol_ma = volume.rolling(VOLUME_RATIO_WINDOW).mean()
        df["volume_ratio"] = volume / (vol_ma + EPS)
    except Exception as exc:  # pragma: no cover
        _logger.error("Failed to compute volume ratio", exc_info=exc)

    # --- ATR ---
    atr = _safe_apply(ta.atr, high, low, close, length=ATR_LENGTH)
    if atr is not None:
        try:
            df["atr_14"] = atr
            df["atr_pct"] = atr / (close + EPS)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute ATR related features", exc_info=exc)

    # --- Stochastic ---
    stoch = _safe_apply(ta.stoch, high, low, close, k=STOCH_K, d=STOCH_D)
    if stoch is not None:
        try:
            df["stoch_k"] = stoch[STOCH_K_COL] / 100.0
            df["stoch_d"] = stoch[STOCH_D_COL] / 100.0
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute Stochastic Oscillator features", exc_info=exc)

    # --- ADX ---
    adx_df = _safe_apply(ta.adx, high, low, close, length=ADX_LENGTH)
    if adx_df is not None:
        try:
            df["adx"] = adx_df[ADX_COL] / 100.0
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute ADX feature", exc_info=exc)

    return df