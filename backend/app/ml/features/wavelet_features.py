"""
Wavelet and spectral feature library — pure numpy/pandas, NO scipy.
All features are computed without lookahead bias (shift(1) applied before adding to df).
Haar DWT implemented manually using [1/√2, 1/√2] and [-1/√2, 1/√2] filters.

Exports:
    add_wavelet_features(df: pd.DataFrame, levels: int = 4) -> pd.DataFrame
    WAVELET_FEATURE_COLS: list[str]
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging & custom exceptions
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

class FeatureComputationError(RuntimeError):
    """Raised when a feature computation fails due to unexpected input or internal error."""
    pass

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
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]
        if len(seg) <= lag:
            continue
        x = seg[:-lag]
        y = seg[lag:]
        mx, my = np.mean(x), np.mean(y)
        sx = np.std(x, ddof=1)
        sy = np.std(y, ddof=1)
        if sx < 1e-12 or sy < 1e-12:
            out[i] = 0.0
        else:
            out[i] = float(np.mean((x - mx) * (y - my)) / (sx * sy))
    return out


# ---------------------------------------------------------------------------
# Statistical moment features (skewness / kurtosis)
# ---------------------------------------------------------------------------

def _rolling_skew(series: np.ndarray, window: int) -> np.ndarray:
    """Rolling sample skewness (Fisher, biased‑corrected denominator n‑1)."""
    if not isinstance(series, np.ndarray):
        raise ValueError("series must be a numpy.ndarray")
    if window < 1:
        raise ValueError("window must be a positive integer")

    n = len(series)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]
        s = np.std(seg, ddof=1)
        if s < 1e-12:
            out[i] = 0.0
        else:
            out[i] = float(np.mean(((seg - np.mean(seg)) / s) ** 3))
    return out


def _rolling_kurt(series: np.ndarray, window: int) -> np.ndarray:
    """Rolling excess kurtosis (Fisher, kurtosis − 3)."""
    if not isinstance(series, np.ndarray):
        raise ValueError("series must be a numpy.ndarray")
    if window < 1:
        raise ValueError("window must be a positive integer")

    n = len(series)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]
        s = np.std(seg, ddof=1)
        if s < 1e-12:
            out[i] = 0.0
        else:
            out[i] = float(np.mean(((seg - np.mean(seg)) / s) ** 4) - 3.0)
    return out


# ---------------------------------------------------------------------------
# Public API – add_wavelet_features
# ---------------------------------------------------------------------------

WAVELET_FEATURE_COLS: List[str] = []  # will be populated by add_wavelet_features


def add_wavelet_features(df: pd.DataFrame, levels: int = 4, window: int = 20) -> pd.DataFrame:
    """
    Compute wavelet‑based energy features for each numeric column in ``df`` and
    return a new DataFrame with the original data plus the generated features.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing one or more numeric time‑series columns.
    levels : int, optional
        Number of Haar decomposition levels (default is 4).
    window : int, optional
        Rolling window size used for all feature calculations (default is 20).

    Returns
    -------
    pd.DataFrame
        DataFrame with the original columns and additional wavelet feature columns.

    Raises
    ------
    FeatureComputationError
        If any internal computation fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas.DataFrame")
    if levels < 1:
        raise ValueError("levels must be a positive integer")
    if window < 1:
        raise ValueError("window must be a positive integer")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        logger.warning("add_wavelet_features called with DataFrame lacking numeric columns")
        return df.copy()

    result = df.copy()
    feature_names: List[str] = []

    for col in numeric_cols:
        series = result[col].to_numpy(dtype=float)

        try:
            dwt_feats = _rolling_dwt_energies(series, window, levels)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to compute DWT energies for column %s", col)
            raise FeatureComputationError(f"DWT energy computation failed for column {col}") from exc

        for lvl in range(levels):
            approx_name = f"{col}_wavelet_approx_l{lvl + 1}"
            detail_name = f"{col}_wavelet_detail_l{lvl + 1}"
            result[approx_name] = dwt_feats[lvl]
            result[detail_name] = dwt_feats[lvl + levels]
            feature_names.extend([approx_name, detail_name])

    # Update module‑level list for external introspection
    global WAVELET_FEATURE_COLS
    WAVELET_FEATURE_COLS = feature_names

    return result