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
Additional composite signals and exit‑related metrics are added to tighten
entry conditions and improve exit logic.
"""

from __future__ import annotations

import logging
import time
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


def _cross_up(series_fast: pd.Series, series_slow: pd.Series) -> pd.Series:
    """
    Detect upward crossovers: fast series moves above slow series.

    Returns a binary series (1 for crossover, 0 otherwise) without look‑ahead.
    """
    crossover = (series_fast > series_slow) & (series_fast.shift(1) <= series_slow.shift(1))
    return crossover.astype(int)


def _cross_down(series_fast: pd.Series, series_slow: pd.Series) -> pd.Series:
    """
    Detect downward crossovers: fast series moves below slow series.

    Returns a binary series (1 for crossover, 0 otherwise) without look‑ahead.
    """
    crossover = (series_fast < series_slow) & (series_fast.shift(1) >= series_slow.shift(1))
    return crossover.astype(int)


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

    start_time = time.perf_counter()
    original_column_count = df.shape[1]

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

    # --- EMA cross signals (fast vs slow) ---
    try:
        ema_fast = close.ewm(span=9).mean()
        ema_slow = close.ewm(span=21).mean()
        df["ema_cross_up"] = _cross_up(ema_fast, ema_slow)
        df["ema_cross_down"] = _cross_down(ema_fast, ema_slow)
    except Exception as exc:  # pragma: no cover
        _logger.error("Failed to compute EMA crossover signals", exc_info=exc)

    # --- RSI ---
    rsi14 = _safe_apply(ta.rsi, close, length=14)
    rsi21 = _safe_apply(ta.rsi, close, length=21)
    if rsi14 is not None:
        df["rsi_14"] = rsi14 / 100.0  # normalize to [0,1]
        df["rsi_14_oversold"] = (df["rsi_14"] < 0.30).astype(int)
        df["rsi_14_overbought"] = (df["rsi_14"] > 0.70).astype(int)
    if rsi21 is not None:
        df["rsi_21"] = rsi21 / 100.0
        df["rsi_21_oversold"] = (df["rsi_21"] < 0.30).astype(int)
        df["rsi_21_overbought"] = (df["rsi_21"] > 0.70).astype(int)

    # --- MACD ---
    macd_df = _safe_apply(ta.macd, close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        try:
            df["macd"] = macd_df["MACD_12_26_9"] / (close + 1e-9)
            df["macd_signal"] = macd_df["MACDs_12_26_9"] / (close + 1e-9)
            df["macd_hist"] = macd_df["MACDh_12_26_9"] / (close + 1e-9)
            df["macd_hist_cross_up"] = _cross_up(df["macd_hist"], pd.Series(0, index=df.index))
            df["macd_hist_cross_down"] = _cross_down(df["macd_hist"], pd.Series(0, index=df.index))
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
            df["bb_breakout_up"] = (close > upper).astype(int)
            df["bb_breakout_down"] = (close < lower).astype(int)
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
            # Exit‑related stop levels (2× ATR)
            df["long_stop"] = close - 2 * atr
            df["short_stop"] = close + 2 * atr
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
            df["trend_strong"] = (df["adx"] > 0.25).astype(int)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed to compute ADX feature", exc_info=exc)

    # --- Composite entry signals ---
    try:
        bullish_conditions = (
            (df.get("ema_cross_up", pd.Series(0, index=df.index)) == 1)
            & (df.get("rsi_14", pd.Series(0, index=df.index)) < 0.30)
            & (df.get("macd_hist", pd.Series(0, index=df.index)) > 0)
            & (df.get("adx", pd.Series(0, index=df.index)) > 0.25)
        )
        bearish_conditions = (
            (df.get("ema_cross_down", pd.Series(0, index=df.index)) == 1)
            & (df.get("rsi_14", pd.Series(0, index=df.index)) > 0.70)
            & (df.get("macd_hist", pd.Series(0, index=df.index)) < 0)
            & (df.get("adx", pd.Series(0, index=df.index)) > 0.25)
        )
        df["bullish_signal"] = bullish_conditions.astype(int)
        df["bearish_signal"] = bearish_conditions.astype(int)
    except Exception as exc:  # pragma: no cover
        _logger.error("Failed to compute composite entry signals", exc_info=exc)

    # Structured logging of key metrics
    new_column_count = df.shape[1]
    signal_count = new_column_count - original_column_count
    execution_time_ms = (time.perf_counter() - start_time) * 1000
    pnl_value = df["pnl"].iloc[-1] if "pnl" in df.columns else None

    _logger.info(
        "Technical feature computation completed",
        extra={
            "signal_count": signal_count,
            "execution_time_ms": execution_time_ms,
            "pnl": pnl_value,
        },
    )

    return df