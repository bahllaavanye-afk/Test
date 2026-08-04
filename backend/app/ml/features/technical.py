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
import time
import unittest
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

import app.ml.features.pandas_ta_compat as ta

_logger = logging.getLogger(__name__)


def _safe_apply(func: Callable[..., pd.DataFrame], *args: Any, **kwargs: Any) -> Optional[pd.DataFrame]:
    """
    Execute a pandas‑ta function safely, catching common errors.

    Parameters
    ----------
    func : Callable[..., pd.DataFrame]
        The pandas‑ta function to invoke.
    *args : Any
        Positional arguments passed to ``func``.
    **kwargs : Any
        Keyword arguments passed to ``func``.

    Returns
    -------
    Optional[pd.DataFrame]
        The result of ``func`` if it succeeds; otherwise ``None`` and an
        error is logged.
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

    The function adds a range of common technical features, normalising
    values where appropriate to keep them on a comparable scale. Features
    that cannot be computed due to missing data or other errors are omitted,
    and the failure is logged.

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
        A new DataFrame containing the original columns plus the computed
        technical feature columns.
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


class TestAddTechnicalFeatures(unittest.TestCase):
    """Edge‑case unit tests for ``add_technical_features``."""

    def test_missing_close_raises(self):
        """Function must raise when the required ``close`` column is absent."""
        df = pd.DataFrame({"high": [1, 2, 3]})
        with self.assertRaises(ValueError):
            add_technical_features(df)

    def test_empty_dataframe_returns_empty(self):
        """An empty DataFrame with a ``close`` column should return an empty result."""
        df = pd.DataFrame({"close": pd.Series([], dtype=float)})
        result = add_technical_features(df)
        self.assertTrue(result.empty)
        self.assertIn("close", result.columns)

    def test_single_row_no_lookahead(self):
        """
        With a single row, look‑ahead dependent features must be NaN.
        Returns_1, volatility, and any rolling calculations should be NaN.
        """
        df = pd.DataFrame(
            {"close": [100.0]},
            index=[pd.Timestamp("2023-01-01")]
        )
        result = add_technical_features(df)

        # Returns should be NaN because there is no previous price
        self.assertTrue(np.isnan(result["returns_1"].iloc[0]))

        # Volatility rolling windows require at least two observations
        self.assertTrue(np.isnan(result["vol_5"].iloc[0]))

        # EMA difference should be zero (close - ema) where ema equals close on first row
        self.assertAlmostEqual(result["ema_9_diff"].iloc[0], 0.0, places=6)

        # Volume ratio should be 1 because rolling mean of a single 1 is 1
        self.assertAlmostEqual(result["volume_ratio"].iloc[0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()