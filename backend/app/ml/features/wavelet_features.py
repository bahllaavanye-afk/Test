"""
Wavelet and spectral feature library — pure numpy/pandas, NO scipy.
All features are computed without lookahead bias (shift(1) applied before adding to df).
Haar DWT implemented manually using [1/√2, 1/√2] and [-1/√2, 1/√2] filters.

Exports:
    add_wavelet_features(df: pd.DataFrame, levels: int = 4, window: int = 64) -> pd.DataFrame
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
    Perform one level of the Haar discrete wavelet transform (DWT) on a 1‑D signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal vector.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (approximation_coeffs, detail_coeffs).  The function pads the input with the
        last value if its length is odd, then computes pair‑wise averages and
        differences using the Haar filters.
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
    signal : np.ndarray
        Input signal vector.
    levels : int
        Number of decomposition levels to compute.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        (approx_list, detail_list) where `approx_list[k]` is the approximation at
        level ``k+1`` and `detail_list[k]` is the corresponding detail coefficients.
        If the signal becomes too short to decompose, zero‑filled arrays are
        returned for the remaining levels.
    """
    approx_levels: List[np.ndarray] = []
    detail_levels: List[np.ndarray] = []
    current = signal.copy()
    for _ in range(levels):
        if len(current) < 2:
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
    """Return the sum of squares of ``arr`` (i.e., its energy)."""
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
    Compute rolling Haar DWT energies for each level.

    Parameters
    ----------
    series : np.ndarray
        Input series (e.g., price or return values).
    window : int
        Rolling window length.
    levels : int
        Number of DWT decomposition levels.

    Returns
    -------
    Tuple[np.ndarray, ...]
        A tuple of ``2 * levels`` arrays, each of length ``len(series)``:
        (approx_l1, …, approx_lN, detail_l1, …, detail_lN).  Positions
        preceding the first full window are ``np.nan``.
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
    Compute rolling spectral features using the Fast Fourier Transform (FFT).

    Parameters
    ----------
    series : np.ndarray
        Input series (e.g., returns or price changes).
    window : int
        Rolling window length.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(spectral_entropy, dominant_freq, power_low, power_mid, power_high)``.
        Each array is length ``len(series)`` with ``np.nan`` for positions
        before the first complete window.  Frequency bands are defined as a
        fraction of the Nyquist frequency:
        * low : ``[0, 0.1)``,
        * mid : ``[0.1, 0.3)``,
        * high: ``[0.3, 1.0]``.
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

        pos_power = power[1:]  # exclude DC component
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
    Compute a rolling Pearson autocorrelation for a given ``lag``.

    Parameters
    ----------
    series : np.ndarray
        Input series.
    lag : int
        Lag (in number of observations) for the autocorrelation.
    window : int
        Rolling window length.

    Returns
    -------
    np.ndarray
        Array of length ``len(series)`` with ``np.nan`` for positions where the
        window is not yet full.  If the standard deviation of either side of the
        lagged pair is close to zero, the correlation is set to ``0.0``.
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
    """
    Compute rolling sample skewness (Fisher definition, biased‑corrected denominator ``n‑1``).

    Parameters
    ----------
    series : np.ndarray
        Input series.
    window : int
        Rolling window length.

    Returns
    -------
    np.ndarray
        Array of rolling skewness values.  Zero is returned when the standard
        deviation within the window is effectively zero.
    """
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
    """
    Compute rolling excess kurtosis (Fisher definition, i.e., kurtosis minus 3).

    Parameters
    ----------
    series : np.ndarray
        Input series.
    window : int
        Rolling window length.

    Returns
    -------
    np.ndarray
        Array of rolling excess kurtosis values.  Zero is returned when the
        standard deviation within the window is effectively zero.
    """
    n = len(series)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]
        s = np.std(seg, ddof=1)
        if s < 1e-12:
            out[i] = 0.0
        else:
            standardized = (seg - np.mean(seg)) / s
            out[i] = float(np.mean(standardized ** 4) - 3.0)
    return out


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def add_wavelet_features(df: pd.DataFrame, levels: int = 4, window: int = 64) -> pd.DataFrame:
    """
    Compute and append wavelet, spectral, and statistical features to ``df``.

    The function expects the DataFrame to contain a column named ``'close'``.
    All rolling calculations are performed on the raw ``close`` price series,
    and the resulting feature columns are added in‑place (a copy of the input
    DataFrame is returned for convenience).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a ``'close'`` column.
    levels : int, optional
        Number of Haar wavelet decomposition levels (default is 4).
    window : int, optional
        Rolling window size for all feature calculations (default is 64).

    Returns
    -------
    pd.DataFrame
        DataFrame that includes the original data plus the newly created feature
        columns.
    """
    if 'close' not in df.columns:
        raise KeyError("Input DataFrame must contain a 'close' column.")

    series = df['close'].astype(float).values

    # Wavelet energies
    dwt_results = _rolling_dwt_energies(series, window, levels)
    approx_energies = dwt_results[:levels]
    detail_energies = dwt_results[levels:]

    for lvl, arr in enumerate(approx_energies, start=1):
        col_name = f'wavelet_approx_energy_l{lvl}'
        df[col_name] = arr

    for lvl, arr in enumerate(detail_energies, start=1):
        col_name = f'wavelet_detail_energy_l{lvl}'
        df[col_name] = arr

    # Spectral features
    spec_entropy, dom_freq, power_low, power_mid, power_high = _spectral_features_rolling(series, window)
    df['spectral_entropy'] = spec_entropy
    df['spectral_dominant_freq'] = dom_freq
    df['spectral_power_low'] = power_low
    df['spectral_power_mid'] = power_mid
    df['spectral_power_high'] = power_high

    # Autocorrelation (lag 1)
    df['autocorr_lag1'] = _rolling_autocorr(series, lag=1, window=window)

    #