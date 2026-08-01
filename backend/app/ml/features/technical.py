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
Additional signal‑quality columns are added to help tighten entry
conditions and provide confirmation filters for downstream strategy logic.
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

    # EMA cross signals (short vs long)
    try:
        ema_short = close.ewm(span=9).mean()
        ema_long = close.ewm(span=21).mean()
        cross_up = (ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1))
        cross_down = (ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1))
        df["ema_cross_up"] = cross_up.astype(int)
        df["ema_cross_down"] = cross_down.astype(int)
    except Exception as exc:  # pragma: no cover
        _logger.error("Failed to compute EMA cross signals", exc_info=exc)

    # --- RSI ---
    rsi14 = _safe_apply(ta.rsi, close, length=14)
    rsi21 = _safe_apply(ta.rsi, close, length=21)
    if rsi14 is not None:
        df["rsi_14"] = rsi14 / 100.0  # normalize to [0,1]
        df["rsi_14_overbought"] = (rsi14 > 70).astype(int)
        df["rsi_14_oversold"] = (rsi14 < 30).astype(int)
    if rsi21 is not None:
        df["rsi_21"] = rsi21 / 100.0
        df["rsi_21_overbought"] = (rsi21 > 70).astype(int)
        df["rsi_21_oversold"] = (rsi21 < 30).astype(int)

    # --- MACD ---
    macd_df = _safe_apply(ta.macd, close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        try:
            df["macd"] = macd_df["MACD_12_26_9"] / (close + 1e-9)
            df["macd_signal"] = macd_df["MACDs_12_26_9"] / (close + 1e-9)
            df["macd_hist"] = macd_df["MACDh_12_26_9"] / (close + 1e-9)

            # MACD histogram crossing zero as a confirmation filter
            macd_cross_up = (macd_df["MACDh_12_26_9"] > 0) & (macd_df["MACDh_12_26_9"].shift(1) <= 0)
            macd_cross_down = (macd_df["MACDh_12_26_9"] < 0) & (macd_df["MACDh_12_26_9"].shift(1) >= 0)
            df["macd_cross_up"] = macd_cross_up.astype(int)
            df["macd_cross_down"] = macd_cross_down.astype(int)
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

            # Bollinger band touch flags
            df["bb_touch_upper"] = (close >= upper * 0.995).astype(int)
            df["bb_touch_lower"] = (close <= lower * 1.005).astype(int)
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

        # Volume spike flag (ratio > 1.5)
        df["volume_spike"] = (df["volume_ratio"] > 1.5).astype(int)
    except Exception as exc:  # pragma: no cover
        _logger.error("Failed to compute volume ratio", exc_info=exc)

    # --- ATR ---
    atr = _safe_apply(ta.atr, high, low, close, length=14)
    if atr is not None:
        try:
            df["atr_14"] = atr
            df["atr_pct"] = atr / (close + 1e-9)

            # Simple ATR‑based stop‑loss levels (long and short)
            df["atr_stop_long"] = close - 2 * atr
            df["atr_stop_short"] = close + 2 * atr
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute ATR related features", exc_info=exc)

    # --- Stochastic ---
    stoch = _safe_apply(ta.stoch, high, low, close, k=14, d=3)
    if stoch is not None:
        try:
            df["stoch_k"] = stoch["STOCHk_14_3_3"] / 100.0
            df["stoch_d"] = stoch["STOCHd_14_3_3"] / 100.0

            # Overbought/oversold flags for stochastic
            df["stoch_overbought"] = (stoch["STOCHk_14_3_3"] > 80).astype(int)
            df["stoch_oversold"] = (stoch["STOCHk_14_3_3"] < 20).astype(int)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute Stochastic Oscillator features", exc_info=exc)

    # --- ADX ---
    adx_df = _safe_apply(ta.adx, high, low, close, length=14)
    if adx_df is not None:
        try:
            df["adx"] = adx_df["ADX_14"] / 100.0

            # ADX strength flag (trend considered strong if > 25)
            df["adx_strong"] = (adx_df["ADX_14"] > 25).astype(int)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute ADX feature", exc_info=exc)

    return df