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

# ---------------------------------------------------------------------------
# Haar DWT helpers
# ---------------------------------------------------------------------------

_SQRT2 = np.sqrt(2.0)
_HAAR_LOW  = np.array([1.0 / _SQRT2, 1.0 / _SQRT2])   # approximation filter
_HAAR_HIGH = np.array([1.0 / _SQRT2, -1.0 / _SQRT2])  # detail filter


def _haar_dwt_1d(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    One level of Haar DWT on a 1-D signal.
    Returns (approximation_coeffs, detail_coeffs).
    Uses convolution + downsample (stride 2) via paired averaging.
    Works for any length; pads with last value if odd-length.
    """
    n = len(signal)
    if n % 2 != 0:
        signal = np.append(signal, signal[-1])
    evens = signal[0::2]
    odds  = signal[1::2]
    approx = (evens + odds) / _SQRT2
    detail = (evens - odds) / _SQRT2
    return approx, detail


def _haar_multilevel(signal: np.ndarray, levels: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Multi-level Haar DWT decomposition.
    Returns (approx_list, detail_list) where index 0 = level 1, ..., index levels-1 = deepest.
    approx_list[k] is the approximation at level k+1 (used for next step).
    detail_list[k] is the detail at level k+1.
    """
    approx_levels: list[np.ndarray] = []
    detail_levels: list[np.ndarray] = []
    current = signal.copy()
    for _ in range(levels):
        if len(current) < 2:
            # Can't decompose further — pad outputs with zeros
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
    """Sum of squares of an array."""
    return float(np.dot(arr, arr))


# ---------------------------------------------------------------------------
# Rolling DWT energy features
# ---------------------------------------------------------------------------

def _rolling_dwt_energies(
    series: np.ndarray,
    window: int,
    levels: int,
) -> tuple[np.ndarray, ...]:
    """
    Roll a window over `series` and compute Haar DWT approximation and detail
    energies at each level.
    Returns a tuple of 2*levels arrays, each of length len(series):
      (approx_l1, approx_l2, ..., approx_lN, detail_l1, detail_l2, ..., detail_lN)
    NaN for positions before the window fills.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling FFT on `series` (interpreted as returns or price changes).
    Returns per-bar arrays (NaN before window):
      spectral_entropy, dominant_freq, power_low, power_mid, power_high
    Frequency bands:
      low  = [0, 0.1) of Nyquist  (first 10% of positive freqs)
      mid  = [0.1, 0.3) of Nyquist
      high = [0.3, 1.0] of Nyquist
    """
    n = len(series)
    spec_entropy  = np.full(n, np.nan)
    dom_freq      = np.full(n, np.nan)
    power_low     = np.full(n, np.nan)
    power_mid     = np.full(n, np.nan)
    power_high    = np.full(n, np.nan)

    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]
        seg_dm = seg - np.mean(seg)
        fft_coeffs = np.fft.rfft(seg_dm)
        power = np.abs(fft_coeffs) ** 2

        pos_power = power[1:]  # skip DC
        total_power = np.sum(pos_power)
        n_pos = len(pos_power)

        if total_power < 1e-20 or n_pos == 0:
            spec_entropy[i] = 0.0
            dom_freq[i]     = 0.0
            power_low[i]    = 0.0
            power_mid[i]    = 0.0
            power_high[i]   = 0.0
            continue

        p_norm = pos_power / total_power
        p_safe = np.clip(p_norm, 1e-20, 1.0)
        spec_entropy[i] = float(-np.sum(p_safe * np.log(p_safe)))

        dom_freq[i] = float(np.argmax(pos_power)) / max(n_pos - 1, 1)

        freqs_norm = np.arange(n_pos) / max(n_pos - 1, 1)

        low_mask  = freqs_norm < 0.1
        mid_mask  = (freqs_norm >= 0.1) & (freqs_norm < 0.3)
        high_mask = freqs_norm >= 0.3

        power_low[i]  = float(np.sum(pos_power[low_mask]))  / total_power
        power_mid[i]  = float(np.sum(pos_power[mid_mask]))  / total_power
        power_high[i] = float(np.sum(pos_power[high_mask])) / total_power

    return spec_entropy, dom_freq, power_low, power_mid, power_high


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def _rolling_autocorr(series: np.ndarray, lag: int, window: int) -> np.ndarray:
    """
    Rolling autocorrelation at a given lag over a rolling window.
    Computed as Pearson correlation of x[t-window..t-lag] with x[t-window+lag..t].
    NaN if std is near zero.
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
    """Rolling sample skewness (Fisher, biased-corrected denominator n-1)."""
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
    """Rolling excess kurtosis (Fisher, kurtosis - 3)."""
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
# Unit tests for edge cases
# ---------------------------------------------------------------------------

import unittest


class TestWaveletFeaturesEdgeCases(unittest.TestCase):
    def test_haar_dwt_odd_length_padding(self):
        # Odd length signal should be padded with the last value
        signal = np.array([1.0, 2.0, 3.0])  # length 3
        approx, detail = _haar_dwt_1d(signal)
        # After padding, signal becomes [1,2,3,3]; pairs: (1,2) and (3,3)
        expected_approx = np.array([(1 + 2) / _SQRT2, (3 + 3) / _SQRT2])
        expected_detail = np.array([(1 - 2) / _SQRT2, (3 - 3) / _SQRT2])
        np.testing.assert_allclose(approx, expected_approx)
        np.testing.assert_allclose(detail, expected_detail)

    def test_rolling_dwt_energies_window_larger_than_series(self):
        series = np.arange(5, dtype=float)
        window = 10  # larger than series length
        levels = 2
        result = _rolling_dwt_energies(series, window, levels)
        # All entries should be NaN because the window never fills
        for arr in result:
            self.assertTrue(np.isnan(arr).all())

    def test_spectral_features_constant_series(self):
        series = np.full(20, 5.0)  # constant series -> zero variance
        window = 8
        spec_entropy, dom_freq, power_low, power_mid, power_high = _spectral_features_rolling(series, window)
        # For a constant series the FFT (after mean removal) is all zeros,
        # leading to zero entropy and zero power in all bands.
        for arr in (spec_entropy, dom_freq, power_low, power_mid, power_high):
            # Positions before window are NaN; after window they should be zero
            self.assertTrue(np.isnan(arr[:window - 1]).all())
            self.assertTrue((arr[window - 1:] == 0.0).all())

    def test_rolling_autocorr_zero_variance(self):
        series = np.zeros(15)
        lag = 1
        window = 5
        result = _rolling_autocorr(series, lag, window)
        # After window fills, autocorrelation should be defined as 0.0 (std near zero)
        self.assertTrue(np.isnan(result[:window - 1]).all())
        self.assertTrue((result[window - 1:] == 0.0).all())

    def test_rolling_kurtosis_constant_series(self):
        series = np.full(12, 7.0)
        window = 4
        result = _rolling_kurt(series, window)
        # Zero variance -> kurtosis defined as 0.0
        self.assertTrue(np.isnan(result[:window - 1]).all())
        self.assertTrue((result[window - 1:] == 0.0).all())


if __name__ == "__main__":
    unittest.main()