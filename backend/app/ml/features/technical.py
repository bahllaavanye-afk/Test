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

Structured logging is added at INFO level to capture key metrics:
- ``signal_count``: number of technical feature columns added.
- ``execution_time_secs``: time taken to compute all features.
- ``pnl``: simple price return proxy (close[-1] / close[0] - 1).
"""

from __future__ import annotations

import logging
import time
from typing import Set

import numpy as np
import pandas as pd

import app.ml.features.pandas_ta_compat as ta

_logger = logging.getLogger(__name__)


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
        feature columns described in the module docstring.

    Notes
    -----
    * All calculations are performed on a copy of ``df`` to avoid mutating
      the original input.
    * Small epsilon values (``1e-9``) are added to denominators to avoid
      division‑by‑zero errors.
    * The function relies on the ``pandas_ta_compat`` wrapper which provides
      a stable API for the underlying ``pandas‑ta`` library.
    """
    start_time = time.perf_counter()
    original_columns: Set[str] = set(df.columns)

    df = df.copy()
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", pd.Series(1, index=df.index))

    # --- Returns ---
    for n in [1, 5, 10, 21]:
        df[f"returns_{n}"] = close.pct_change(n)

    # --- Volatility (rolling std of log returns) ---
    log_ret = np.log(close / close.shift(1))
    for n in [5, 21, 63]:
        df[f"vol_{n}"] = log_ret.rolling(n).std() * np.sqrt(252)

    # --- EMA distance (normalized) ---
    for span in [9, 21, 50]:
        ema = close.ewm(span=span).mean()
        df[f"ema_{span}_diff"] = (close - ema) / (ema + 1e-9)

    # --- RSI ---
    rsi14 = ta.rsi(close, length=14)
    rsi21 = ta.rsi(close, length=21)
    if rsi14 is not None:
        df["rsi_14"] = rsi14 / 100.0  # normalize to [0,1]
    if rsi21 is not None:
        df["rsi_21"] = rsi21 / 100.0

    # --- MACD ---
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        df["macd"] = macd_df["MACD_12_26_9"] / (close + 1e-9)
        df["macd_signal"] = macd_df["MACDs_12_26_9"] / (close + 1e-9)
        df["macd_hist"] = macd_df["MACDh_12_26_9"] / (close + 1e-9)

    # --- Bollinger Bands ---
    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None:
        upper = bb["BBU_20_2.0"]
        lower = bb["BBL_20_2.0"]
        mid = bb["BBM_20_2.0"]
        df["bb_upper_dist"] = (upper - close) / (close + 1e-9)
        df["bb_lower_dist"] = (close - lower) / (close + 1e-9)
        df["bb_width"] = (upper - lower) / (mid + 1e-9)

    # --- OBV change (normalized) ---
    obv = ta.obv(close, volume)
    if obv is not None:
        df["obv_change"] = obv.pct_change(5).fillna(0)

    # --- Volume ratio ---
    vol_ma = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (vol_ma + 1e-9)

    # --- ATR ---
    atr = ta.atr(high, low, close, length=14)
    if atr is not None:
        df["atr_14"] = atr
        df["atr_pct"] = atr / (close + 1e-9)

    # --- Stochastic ---
    stoch = ta.stoch(high, low, close, k=14, d=3)
    if stoch is not None:
        df["stoch_k"] = stoch["STOCHk_14_3_3"] / 100.0
        df["stoch_d"] = stoch["STOCHd_14_3_3"] / 100.0

    # --- ADX ---
    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None:
        df["adx"] = adx_df["ADX_14"] / 100.0

    # Metrics for logging
    new_columns = set(df.columns) - original_columns
    signal_count = len(new_columns)
    execution_time_secs = time.perf_counter() - start_time
    # Simple P&L proxy: total return of the close series
    if len(close) > 1 and close.iloc[0] != 0:
        pnl = float(close.iloc[-1] / close.iloc[0] - 1)
    else:
        pnl = 0.0

    _logger.info(
        "Technical features added",
        extra={
            "signal_count": signal_count,
            "execution_time_secs": execution_time_secs,
            "pnl": pnl,
        },
    )

    return df