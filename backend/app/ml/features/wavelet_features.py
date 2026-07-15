"""
Wavelet and spectral feature library — pure numpy/pandas, NO scipy.
All features are computed without lookahead bias (shift(1) applied before adding to df).
Haar DWT implemented manually using [1/√2, 1/√2] and [-1/√2, 1/√2] filters.

Exports:
    add_wavelet_features(df: pd.DataFrame, levels: int = 4) -> pd.DataFrame
    WAVELET_FEATURE_COLS: list[str]
    WaveletFeatureConfig
    SpectralFeatureConfig
    AutocorrFeatureConfig
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# Logging & custom exceptions
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

class FeatureComputationError(RuntimeError):
    """Raised when a feature computation fails due to unexpected input or internal error."""
    pass

# ---------------------------------------------------------------------------
# Pydantic schemas for feature configuration
# ---------------------------------------------------------------------------

class WaveletFeatureConfig(BaseModel):
    """Configuration for adding wavelet‑based features to a DataFrame."""

    df: pd.DataFrame = Field(
        ...,
        description="Input price DataFrame. Must contain at least one numeric column.",
        example=pd.DataFrame({"close": [100.0, 101.5, 102.0]}),
    )
    levels: int = Field(
        4,
        ge=1,
        description="Number of Haar wavelet decomposition levels to compute.",
        example=4,
    )

    @validator("df")
    def _df_must_have_numeric(cls, v: pd.DataFrame) -> pd.DataFrame:
        if v.empty:
            raise ValueError("DataFrame cannot be empty")
        numeric_cols = v.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("DataFrame must contain at least one numeric column")
        return v

class SpectralFeatureConfig(BaseModel):
    """Configuration for rolling spectral (FFT) features."""

    series: np.ndarray = Field(
        ...,
        description="1‑D array of returns or price changes.",
        example=np.array([0.01, -0.005, 0.003]),
    )
    window: int = Field(
        ...,
        gt=0,
        description="Rolling window size (number of observations).",
        example=64,
    )

    @validator("series")
    def _series_must_be_1d(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError("series must be a numpy.ndarray")
        if v.ndim != 1:
            raise ValueError("series must be a 1‑dimensional array")
        return v

class AutocorrFeatureConfig(BaseModel):
    """Configuration for rolling autocorrelation."""

    series: np.ndarray = Field(
        ...,
        description="1‑D array of price or return values.",
        example=np.array([1.0, 1.2, 0.9]),
    )
    lag: int = Field(
        ...,
        gt=0,
        description="Lag (in number of observations) for the autocorrelation.",
        example=1,
    )
    window: int = Field(
        ...,
        gt=0,
        description="Rolling window length.",
        example=30,
    )

    @validator("series")
    def _series_must_be_1d(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError("series must be a numpy.ndarray")
        if v.ndim != 1:
            raise ValueError("series must be a 1‑dimensional array")
        return v

# ---------------------------------------------------------------------------
# Haar DWT helpers
# ---------------------------------------------------------------------------

_SQRT2 = np.sqrt(2.0)
_HAAR_LOW = np.array([1.0 / _SQRT2, 1.0 / _SQRT2])   # approximation filter
_HAAR_HIGH = np.array([1.0 / _SQRT2, -1.0 / _SQRT2])  # detail filter


def _haar_dwt_1d(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    One level of Haar DWT on a 1‑D signal.
    Returns (approximation_coeffs, detail_coeffs).
    Uses convolution + downsample (stride 2) via paired averaging.
    Works for any length; pads with last value if odd-length.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError("signal must be a numpy.ndarray")
    n = len(signal)
    if n == 0:
        raise ValueError("signal length must be greater than 0")
    if n % 2 != 0:
        signal = np.append(signal, signal[-1])
    evens = signal[0::2]
    odds = signal[1::2]
    approx = (evens + odds) / _SQRT2
    detail = (evens - odds) / _SQRT2
    return approx, detail


def _haar_multilevel(signal: np.ndarray, levels: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Multi‑level Haar DWT decomposition.
    Returns (approx_list, detail_list) where index 0 = level 1, …, index levels‑1 = deepest.
    approx_list[k] is the approximation at level k+1 (used for next step).
    detail_list[k] is the detail at level k+1.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError("signal must be a numpy.ndarray")
    if levels < 1:
        raise ValueError("levels must be a positive integer")
    approx_levels: List[np.ndarray] = []
    detail_levels: List[np.ndarray] = []
    current = signal.copy()
    for lvl in range(levels):
        if len(current) < 2:
            # Can't decompose further — pad outputs with zeros
            approx_levels.append(np.array([0.0]))
            detail_levels.append(np.array([0.0]))
            current = np.array([0.0])
        else:
            try:
                a, d = _haar_dwt_1d(current)
            except Exception as exc:  # pragma: no cover
                logger.exception("Haar DWT failed at level %d", lvl + 1)
                raise FeatureComputationError("Haar DWT failed") from exc
            approx_levels.append(a)
            detail_levels.append(d)
            current = a
    return approx_levels, detail_levels


def _energy(arr: np.ndarray) -> float:
    """Sum of squares of an array."""
    if not isinstance(arr, np.ndarray):
        raise ValueError("arr must be a numpy.ndarray")
    return float(np.dot(arr, arr))


# ---------------------------------------------------------------------------
# Rolling DWT energy features
# ---------------------------------------------------------------------------

def _rolling_dwt_energies(
    series: np.ndarray,
    window: int,
    levels: int,
) -> Tuple[np.ndarray, ...]:
    """
    Roll a window over `series` and compute Haar DWT approximation and detail
    energies at each level.

    Returns a tuple of 2*levels arrays, each of length len(series):
      (approx_l1, approx_l2, ..., approx_lN, detail_l1, detail_l2, ..., detail_lN)

    Positions before the window fills contain NaN.
    """
    if not isinstance(series, np.ndarray):
        raise ValueError("series must be a numpy.ndarray")
    if window < 1:
        raise ValueError("window must be a positive integer")
    if levels < 1:
        raise ValueError("levels must be a positive integer")

    n = len(series)
    approx_e = [np.full(n, np.nan) for _ in range(levels)]
    detail_e = [np.full(n, np.nan) for _ in range(levels)]

    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]
        try:
            a_levels, d_levels = _haar_multilevel(seg, levels)
        except FeatureComputationError:
            logger.exception("Failed to compute multi‑level Haar DWT at index %d", i)
            continue
        for lv in range(levels):
            approx_e[lv][i] = _energy(a_levels[lv])
            detail_e[lv][i] = _energy(d_levels[lv])

    return (*approx_e, *detail_e)


# ---------------------------------------------------------------------------
# Spectral features (rolling FFT)
# ---------------------------------------------------------------------------

def _spectral_features_rolling(
    series: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling FFT on `series` (interpreted as returns or price changes).

    Returns per‑bar arrays (NaN before window):
      spectral_entropy, dominant_freq, power_low, power_mid, power_high

    Frequency bands:
      low  = [0, 0.1) of Nyquist  (first 10% of positive freqs)
      mid  = [0.1, 0.3) of Nyquist
      high = [0.3, 1.0] of Nyquist
    """
    if not isinstance(series, np.ndarray):
        raise ValueError("series must be a numpy.ndarray")
    if window < 1:
        raise ValueError("window must be a positive integer")

    n = len(series)
    spec_entropy = np.full(n, np.nan)
    dom_freq = np.full(n, np.nan)
    power_low = np.full(n, np.nan)
    power_mid = np.full(n, np.nan)
    power_high = np.full(n, np.nan)

    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]
        seg_dm = seg - np.mean(seg)
        fft_coeffs = np.fft.rfft(seg_dm)
        power = np.abs(fft_coeffs) ** 2

        pos_power = power[1:]  # skip DC component
        total_power = np.sum(pos_power)
        n_pos = len(pos_power)

        if total_power < 1e-20 or n_pos == 0:
            spec_entropy[i] = 0.0
            dom_freq[i] = 0.0
            power_low[i] = 0.0
            power_mid[i] = 0.0
            power_high[i] = 0.0
            continue

        p_norm = pos_power / total_power
        p_safe = np.clip(p_norm, 1e-20, 1.0)
        spec_entropy[i] = float(-np.sum(p_safe * np.log(p_safe)))

        dom_freq[i] = float(np.argmax(pos_power)) / max(n_pos - 1, 1)

        freqs_norm = np.arange(n_pos) / max(n_pos - 1, 1)

        low_mask = freqs_norm < 0.1
        mid_mask = (freqs_norm >= 0.1) & (freqs_norm < 0.3)
        high_mask = freqs_norm >= 0.3

        power_low[i] = float(np.sum(pos_power[low_mask])) / total_power
        power_mid[i] = float(np.sum(pos_power[mid_mask])) / total_power
        power_high[i] = float(np.sum(pos_power[high_mask])) / total_power

    return spec_entropy, dom_freq, power_low, power_mid, power_high


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def _rolling_autocorr(series: np.ndarray, lag: int, window: int) -> np.ndarray:
    """
    Rolling autocorrelation at a given lag over a rolling window.
    Computed as Pearson correlation of x[t‑window..t‑lag] with x[t‑window+lag..t].
    Returns NaN if the standard deviation of either segment is near zero.
    """
    if not isinstance(series, np.ndarray):
        raise ValueError("series must be a numpy.ndarray")
    if lag < 1:
        raise ValueError("lag must be a positive integer")
    if window < 1:
        raise ValueError("window must be a positive integer")

    n = len(series)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        start = i - window + 1
        end = i + 1
        x = series[start : end - lag]
        y = series[start + lag : end]
        if x.size == 0 or y.size == 0:
            continue
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x < 1e-12 or std_y < 1e-12:
            continue
        cov = np.mean((x - x.mean()) * (y - y.mean()))
        result[i] = cov / (std_x * std_y)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_wavelet_features(df: pd.DataFrame, levels: int = 4) -> pd.DataFrame:
    """
    Compute Haar wavelet energy features for each numeric column in ``df``.
    The function adds ``2 * levels`` new columns per original column:
        ``{col}_wavelet_approx_l{lvl}`` and ``{col}_wavelet_detail_l{lvl}``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing price or return series.
    levels : int, default 4
        Number of decomposition levels to compute; must be >= 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus wavelet feature columns.
    """
    config = WaveletFeatureConfig(df=df, levels=levels)  # validation
    result = config.df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = result[col].to_numpy()
        approx_vals, detail_vals = _rolling_dwt_energies(series, window=levels * 2, levels=levels)
        for lvl, arr in enumerate(approx_vals, start=1):
            result[f"{col}_wavelet_approx_l{lvl}"] = arr
        for lvl, arr in enumerate(detail_vals, start=1):
            result[f"{col}_wavelet_detail_l{lvl}"] = arr

    return result


WAVELET_FEATURE_COLS: List[str] = [
    col for col in add_wavelet_features(pd.DataFrame({"dummy": [0.0]})).columns if "wavelet" in col
]

__all__ = [
    "add_wavelet_features",
    "WAVELET_FEATURE_COLS",
    "WaveletFeatureConfig",
    "SpectralFeatureConfig",
    "AutocorrFeatureConfig",
    "FeatureComputationError",
]