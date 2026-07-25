from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

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

    return df


class TechnicalFeatureConfig(BaseModel):
    """
    Configuration schema for technical feature generation.

    This schema centralises all tunable parameters used by
    :func:`add_technical_features`. It provides validation, documentation,
    and example values to aid developers and API consumers.
    """

    returns_periods: list[int] = Field(
        default_factory=lambda: [1, 5, 10, 21],
        description="Look‑back periods (in bars) for simple return calculations.",
        example=[1, 5, 10, 21],
    )
    volatility_periods: list[int] = Field(
        default_factory=lambda: [5, 21, 63],
        description="Rolling window sizes (in bars) for volatility (standard deviation of log returns).",
        example=[5, 21, 63],
    )
    ema_spans: list[int] = Field(
        default_factory=lambda: [9, 21, 50],
        description="Span values for Exponential Moving Average distance features.",
        example=[9, 21, 50],
    )
    rsi_lengths: list[int] = Field(
        default_factory=lambda: [14, 21],
        description="Lengths for Relative Strength Index calculations.",
        example=[14, 21],
    )
    macd_fast: int = Field(
        default=12,
        description="Fast EMA period for MACD.",
        example=12,
        ge=1,
    )
    macd_slow: int = Field(
        default=26,
        description="Slow EMA period for MACD.",
        example=26,
        ge=1,
    )
    macd_signal: int = Field(
        default=9,
        description="Signal line EMA period for MACD.",
        example=9,
        ge=1,
    )
    bb_length: int = Field(
        default=20,
        description="Look‑back period for Bollinger Bands.",
        example=20,
        ge=1,
    )
    bb_std: float = Field(
        default=2.0,
        description="Number of standard deviations for Bollinger Bands width.",
        example=2.0,
        gt=0.0,
    )
    volume_ma_window: int = Field(
        default=20,
        description="Window size for moving average of volume used in volume ratio feature.",
        example=20,
        ge=1,
    )
    atr_length: int = Field(
        default=14,
        description="Look‑back period for Average True Range.",
        example=14,
        ge=1,
    )
    stoch_k: int = Field(
        default=14,
        description="Look‑back period for %K line of Stochastic Oscillator.",
        example=14,
        ge=1,
    )
    stoch_d: int = Field(
        default=3,
        description="Smoothing period for %D line of Stochastic Oscillator.",
        example=3,
        ge=1,
    )
    adx_length: int = Field(
        default=14,
        description="Look‑back period for Average Directional Index.",
        example=14,
        ge=1,
    )

    @validator(
        "returns_periods",
        "volatility_periods",
        "ema_spans",
        "rsi_lengths",
        each_item=True,
    )
    def positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("All period values must be positive integers.")
        return v

    @validator("bb_std")
    def positive_float(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("bb_std must be a positive float.")
        return v


__all__ = ["add_technical_features", "TechnicalFeatureConfig"]