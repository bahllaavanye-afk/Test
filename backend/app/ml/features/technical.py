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
from typing import Any

import numpy as np
import pandas as pd

import app.ml.features.pandas_ta_compat as ta

_logger = logging.getLogger(__name__)


def _safe_apply(func: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Helper to execute a function safely, logging any exception and returning ``None``.

    Parameters
    ----------
    func : Callable
        The function to execute.
    *args, **kwargs
        Arguments passed to ``func``.

    Returns
    -------
    Any
        The result of ``func`` if successful; otherwise ``None``.
    """
    try:
        return func(*args, **kwargs)
    except (ValueError, TypeError, KeyError) as exc:
        _logger.exception("Error in %s with args=%s, kwargs=%s", func.__name__, args, kwargs)
        return None
    except Exception as exc:  # pragma: no cover
        _logger.exception("Unexpected error in %s", func.__name__)
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
        A new DataFrame containing the original columns plus the following
        technical feature columns:

        - ``returns_{n}`` : Percentage change over *n* periods for n in
          ``[1, 5, 10, 21]``.
        - ``vol_{n}`` : Annualised volatility (rolling std of log returns)
          for n in ``[5, 21, 63]``.
        - ``ema_{span}_diff`` : Normalised distance between price and its EMA
          for spans 9, 21, 50.
        - ``rsi_14``, ``rsi_21`` : Normalised RSI values.
        - ``macd``, ``macd_signal``, ``macd_hist`` : Normalised MACD components.
        - ``bb_upper_dist``, ``bb_lower_dist``, ``bb_width`` : Bollinger
          Band distances and width.
        - ``obv_change`` : 5‑period percentage change of On‑Balance Volume.
        - ``volume_ratio`` : Current volume divided by its 20‑period moving
          average.
        - ``atr_14``, ``atr_pct`` : Average True Range and its percentage of
          price.
        - ``stoch_k``, ``stoch_d`` : Normalised Stochastic %K and %D.
        - ``adx`` : Normalised Average Directional Index.

    Notes
    -----
    * All calculations are performed on a copy of ``df`` to avoid mutating
      the original input.
    * Small epsilon values (``1e-9``) are added to denominators to avoid
      division‑by‑zero errors.
    * The function relies on the ``pandas_ta_compat`` wrapper which provides
      a stable API for the underlying ``pandas‑ta`` library.
    """
    if "close" not in df.columns:
        _logger.error("Input DataFrame missing required 'close' column")
        raise KeyError("Input DataFrame must contain a 'close' column")

    df = df.copy()
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", pd.Series(1, index=df.index))

    # --- Returns ---
    for n in [1, 5, 10, 21]:
        try:
            df[f"returns_{n}"] = close.pct_change(n)
        except Exception:
            _logger.exception("Failed to compute returns for period %s", n)
            df[f"returns_{n}"] = np.nan

    # --- Volatility (rolling std of log returns) ---
    try:
        log_ret = np.log(close / close.shift(1))
        for n in [5, 21, 63]:
            df[f"vol_{n}"] = log_ret.rolling(n).std() * np.sqrt(252)
    except Exception:
        _logger.exception("Failed to compute volatility")
        for n in [5, 21, 63]:
            df[f"vol_{n}"] = np.nan

    # --- EMA distance (normalized) ---
    for span in [9, 21, 50]:
        try:
            ema = close.ewm(span=span).mean()
            df[f"ema_{span}_diff"] = (close - ema) / (ema + 1e-9)
        except Exception:
            _logger.exception("Failed to compute EMA distance for span %s", span)
            df[f"ema_{span}_diff"] = np.nan

    # --- RSI ---
    rsi14 = _safe_apply(ta.rsi, close, length=14)
    rsi21 = _safe_apply(ta.rsi, close, length=21)
    if rsi14 is not None:
        df["rsi_14"] = rsi14 / 100.0  # normalize to [0,1]
    else:
        df["rsi_14"] = np.nan
    if rsi21 is not None:
        df["rsi_21"] = rsi21 / 100.0
    else:
        df["rsi_21"] = np.nan

    # --- MACD ---
    macd_df = _safe_apply(ta.macd, close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        try:
            df["macd"] = macd_df["MACD_12_26_9"] / (close + 1e-9)
            df["macd_signal"] = macd_df["MACDs_12_26_9"] / (close + 1e-9)
            df["macd_hist"] = macd_df["MACDh_12_26_9"] / (close + 1e-9)
        except Exception:
            _logger.exception("Failed to normalize MACD components")
            df["macd"] = df["macd_signal"] = df["macd_hist"] = np.nan
    else:
        df["macd"] = df["macd_signal"] = df["macd_hist"] = np.nan

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
        except Exception:
            _logger.exception("Failed to compute Bollinger Band features")
            df["bb_upper_dist"] = df["bb_lower_dist"] = df["bb_width"] = np.nan
    else:
        df["bb_upper_dist"] = df["bb_lower_dist"] = df["bb_width"] = np.nan

    # --- OBV change (normalized) ---
    obv = _safe_apply(ta.obv, close, volume)
    if obv is not None:
        try:
            df["obv_change"] = obv.pct_change(5).fillna(0)
        except Exception:
            _logger.exception("Failed to compute OBV change")
            df["obv_change"] = np.nan
    else:
        df["obv_change"] = np.nan

    # --- Volume ratio ---
    try:
        vol_ma = volume.rolling(20).mean()
        df["volume_ratio"] = volume / (vol_ma + 1e-9)
    except Exception:
        _logger.exception("Failed to compute volume ratio")
        df["volume_ratio"] = np.nan

    # --- ATR ---
    atr = _safe_apply(ta.atr, high, low, close, length=14)
    if atr is not None:
        try:
            df["atr_14"] = atr
            df["atr_pct"] = atr / (close + 1e-9)
        except Exception:
            _logger.exception("Failed to normalize ATR")
            df["atr_14"] = df["atr_pct"] = np.nan
    else:
        df["atr_14"] = df["atr_pct"] = np.nan

    # --- Stochastic ---
    stoch = _safe_apply(ta.stoch, high, low, close, k=14, d=3)
    if stoch is not None:
        try:
            df["stoch_k"] = stoch["STOCHk_14_3_3"] / 100.0
            df["stoch_d"] = stoch["STOCHd_14_3_3"] / 100.0
        except Exception:
            _logger.exception("Failed to compute Stochastic oscillator")
            df["stoch_k"] = df["stoch_d"] = np.nan
    else:
        df["stoch_k"] = df["stoch_d"] = np.nan

    # --- ADX ---
    adx_df = _safe_apply(ta.adx, high, low, close, length=14)
    if adx_df is not None:
        try:
            df["adx"] = adx_df["ADX_14"] / 100.0
        except Exception:
            _logger.exception("Failed to normalize ADX")
            df["adx"] = np.nan
    else:
        df["adx"] = np.nan

    return df