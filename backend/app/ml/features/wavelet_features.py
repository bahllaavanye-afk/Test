"""
Wavelet and spectral feature library — pure numpy/pandas, NO scipy.
All features are computed without lookahead bias (shift(1) applied before adding to df).
Haar DWT implemented manually using [1/√2, 1/√2] and [-1/√2, 1/√2] filters.

Exports:
    add_wavelet_features(df: pd.DataFrame, levels: int = 4) -> pd.DataFrame
    WAVELET_FEATURE_COLS: list[str]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Haar DWT helpers
# ---------------------------------------------------------------------------

_SQRT2 = np.sqrt(2.0)
_HAAR_LOW = np.array([1.0 / _SQRT2, 1.0 / _SQRT2])   # approximation filter
_HAAR_HIGH = np.array([1.0 / _SQRT2, -1.0 / _SQRT2])  # detail filter


def _haar_dwt_1d(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one level of Haar Discrete Wavelet Transform on a 1‑D signal.

    Parameters
    ----------
    signal: np.ndarray
        Input signal. If its length is odd it will be padded with the last value.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (approximation_coeffs, detail_coeffs) for the given level.
    """
    n = len(signal)
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

    Parameters
    ----------
    signal: np.ndarray
        Input signal.
    levels: int
        Number of decomposition levels.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        (approx_list, detail_list) where index 0 corresponds to level 1.
        ``approx_list[k]`` is the approximation at level ``k+1`` (used for the
        next decomposition step). ``detail_list[k]`` is the detail coefficients
        at level ``k+1``.
    """
    approx_levels: List[np.ndarray] = []
    detail_levels: List[np.ndarray] = []
    current = signal.copy()
    for _ in range(levels):
        if len(current) < 2:
            # Cannot decompose further – pad with zeros to keep shape consistent
            approx_levels.append(np.array([0.0]))
            detail_levels.append(np.array([0.0]))
            current = np.array([0.0])
        else:
            a, d = _haar_dwt_1d(current)
            approx_levels.append(a)
            detail_levels.append(d)
            current = a
    return approx_levels, detail_levels


def _energy(arr: np.ndarray) -> float:
    """Return the sum of squares of ``arr`` (i.e. its L2 energy)."""
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
    Compute rolling Haar DWT energies for a series.

    Parameters
    ----------
    series: np.ndarray
        Input time‑series.
    window: int
        Rolling window length.
    levels: int
        Number of decomposition levels.

    Returns
    -------
    Tuple[np.ndarray, ...]
        A tuple of ``2 * levels`` arrays, each of length ``len(series)``:
        ``(approx_l1, …, approx_lN, detail_l1, …, detail_lN)``.
        Positions before the window is filled contain ``np.nan``.
    """
    n = len(series)
    approx_e = [np.full(n, np.nan) for _ in range(levels)]
    detail_e = [np.full(n, np.nan) for _ in range(levels)]

    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]
        a_levels, d_levels = _haar_multilevel(seg, levels)
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
    Compute rolling spectral features using FFT.

    Parameters
    ----------
    series: np.ndarray
        Input time‑series (e.g., returns or price changes).
    window: int
        Rolling window length.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(spectral_entropy, dominant_freq, power_low, power_mid, power_high)``.
        All arrays have length ``len(series)`` and contain ``np.nan`` before the
        window is filled.
    """
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

        pos_power = power[1:]  # discard DC component
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
    Compute rolling autocorrelation for a given ``lag`` over a sliding window.

    Parameters
    ----------
    series: np.ndarray
        Input time‑series.
    lag: int
        Lag at which to compute the autocorrelation.
    window: int
        Rolling window length.

    Returns
    -------
    np.ndarray
        Array of autocorrelation values; ``np.nan`` where the window is not
        fully populated.
    """
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
    """Rolling sample skewness (Fisher, biased‑corrected denominator *n‑1*)."""
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
    """Rolling excess kurtosis (Fisher, i.e. kurtosis − 3)."""
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
# Public API – feature aggregation
# ---------------------------------------------------------------------------

def add_wavelet_features(df: pd.DataFrame, levels: int = 4) -> pd.DataFrame:
    """
    Append wavelet‑based and spectral features to ``df``.

    The function operates on the first numeric column of ``df`` (typically a
    price or return series).  A fixed rolling window of 32 observations is used.
    All computed features are shifted by one period to avoid look‑ahead bias.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame. Must contain at least one numeric column.
    levels: int, default=4
        Number of Haar DWT decomposition levels.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional feature columns.
    """
    if df.empty:
        return df

    # Identify the first numeric column to use as the source series.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame must contain at least one numeric column.")
    source_col = numeric_cols[0]
    series = df[source_col].to_numpy(dtype=float)

    window = 32  # a reasonable default for intraday / daily data

    # Compute rolling DWT energies.
    dwt_arrays = _rolling_dwt_energies(series, window, levels)

    # Compute spectral features.
    spec_entropy, dom_freq, power_low, power_mid, power_high = _spectral_features_rolling(series, window)

    # Compute additional statistical features.
    autocorr_lag1 = _rolling_autocorr(series, lag=1, window=window)
    skew = _rolling_skew(series, window)
    kurt = _rolling_kurt(series, window)

    # Shift everything by one period to ensure no look‑ahead bias.
    shift = 1
    feature_data: dict[str, np.ndarray] = {}

    for lvl, arr in enumerate(dwt_arrays[:levels], start=1):
        feature_data[f"wavelet_approx_l{lvl}"] = np.roll(arr, shift)
    for lvl, arr in enumerate(dwt_arrays[levels:], start=1):
        feature_data[f"wavelet_detail_l{lvl}"] = np.roll(arr, shift)

    feature_data["spectral_entropy"] = np.roll(spec_entropy, shift)
    feature_data["spectral_dom_freq"] = np.roll(dom_freq, shift)
    feature_data["spectral_power_low"] = np.roll(power_low, shift)
    feature_data["spectral_power_mid"] = np.roll(power_mid, shift)
    feature_data["spectral_power_high"] = np.roll(power_high, shift)
    feature_data["autocorr_lag1"] = np.roll(autocorr_lag1, shift)
    feature_data["skewness"] = np.roll(skew, shift)
    feature_data["excess_kurtosis"] = np.roll(kurt, shift)

    for col_name, values in feature_data.items():
        df[col_name] = values

    return df


# List of columns added by ``add_wavelet_features`` – useful for downstream pipelines.
WAVELET_FEATURE_COLS: List[str] = [
    "wavelet_approx_l1",
    "wavelet_approx_l2",
    "wavelet_approx_l3",
    "wavelet_approx_l4",
    "wavelet_detail_l1",
    "wavelet_detail_l2",
    "wavelet_detail_l3",
    "wavelet_detail_l4",
    "spectral_entropy",
    "spectral_dom_freq",
    "spectral_power_low",
    "spectral_power_mid",
    "spectral_power_high",
    "autocorr_lag1",
    "skewness",
    "excess_kurtosis",
]

__all__ = ["add_wavelet_features", "WAVELET_FEATURE_COLS"]