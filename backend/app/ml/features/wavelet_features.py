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
    # Pair-wise operations (equivalent to convolution with Haar filters + stride 2)
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
        # Remove mean before FFT (DC component carries no spectral info)
        seg_dm = seg - np.mean(seg)
        fft_coeffs = np.fft.rfft(seg_dm)
        power = np.abs(fft_coeffs) ** 2

        # Only positive frequencies (rfft gives [0 .. Nyquist])
        # Index 0 is DC, skip it to avoid DC dominating entropy
        pos_power = power[1:]  # length = window//2

        total_power = np.sum(pos_power)
        n_pos = len(pos_power)

        if total_power < 1e-20 or n_pos == 0:
            spec_entropy[i] = 0.0
            dom_freq[i]     = 0.0
            power_low[i]    = 0.0
            power_mid[i]    = 0.0
            power_high[i]   = 0.0
            continue

        # Normalized power spectrum (probability distribution)
        p_norm = pos_power / total_power

        # Spectral entropy: -sum(p * log(p)), clipped to avoid log(0)
        p_safe = np.clip(p_norm, 1e-20, 1.0)
        spec_entropy[i] = float(-np.sum(p_safe * np.log(p_safe)))

        # Dominant frequency index, normalized to [0, 1]
        dom_freq[i] = float(np.argmax(pos_power)) / max(n_pos - 1, 1)

        # Band power fractions
        # freq index relative to n_pos gives normalized frequency in [0,1]
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
            out[i] = float(np.mean(((seg - np.mean(seg)) / s) ** 4)) - 3.0
    return out


# ---------------------------------------------------------------------------
# Cross-correlation between price returns and volume changes
# ---------------------------------------------------------------------------

def _rolling_xcorr(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
    window: int,
) -> np.ndarray:
    """
    Rolling cross-correlation between x and y at the given lag.
    y is lagged by `lag` bars relative to x.
    """
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        xs = x[i - window + 1 : i + 1]
        ys = y[i - window + 1 : i + 1]
        if lag > 0:
            xa = xs[lag:]
            ya = ys[:-lag]
        else:
            xa = xs
            ya = ys
        if len(xa) < 2:
            continue
        sx = np.std(xa, ddof=1)
        sy = np.std(ya, ddof=1)
        if sx < 1e-12 or sy < 1e-12:
            out[i] = 0.0
        else:
            out[i] = float(np.mean((xa - np.mean(xa)) * (ya - np.mean(ya))) / (sx * sy))
    return out


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def add_wavelet_features(df: pd.DataFrame, levels: int = 4) -> pd.DataFrame:
    """
    Compute Haar DWT-based and spectral/statistical features and add them to df.

    Expects df to have columns: close, volume.
    All features are shifted by 1 bar before assignment to prevent lookahead bias.

    Args:
        df:     OHLCV DataFrame (at minimum close and volume required).
        levels: Number of DWT decomposition levels (default 4).

    Returns:
        DataFrame with WAVELET_FEATURE_COLS columns appended.
    """
    df = df.copy()

    close   = df["close"].values.astype(float)
    volume  = df["volume"].values.astype(float)
    returns = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=np.nan)
    vol_chg = np.diff(np.log(np.clip(volume, 1e-12, None)), prepend=np.nan)

    n = len(close)

    # Window sizes used for rolling calculations
    dwt_window      = 64   # must be >= 2**levels for meaningful decomposition
    fft_window      = 64
    autocorr_window = 63
    moment_window   = 21
    xcorr_window    = 63

    # -----------------------------------------------------------------------
    # 1-8: DWT approximation and detail energies at levels 1..levels
    # -----------------------------------------------------------------------
    dwt_arrays = _rolling_dwt_energies(close, window=dwt_window, levels=levels)
    # dwt_arrays layout: (approx_l1..approx_lN, detail_l1..detail_lN)
    for lv in range(levels):
        col_approx = f"dwt_approx_l{lv + 1}_energy"
        col_detail  = f"dwt_detail_l{lv + 1}_energy"
        # Shift 1 to prevent lookahead
        df[col_approx] = pd.Series(dwt_arrays[lv],           index=df.index).shift(1)
        df[col_detail]  = pd.Series(dwt_arrays[levels + lv], index=df.index).shift(1)

    # -----------------------------------------------------------------------
    # 9: DWT noise ratio: detail_l1_energy / (detail_lN_energy + eps)
    # -----------------------------------------------------------------------
    detail_l1 = dwt_arrays[levels + 0]  # level 1 detail energy
    detail_lN = dwt_arrays[levels + levels - 1]  # deepest level detail energy
    noise_ratio_raw = detail_l1 / (detail_lN + 1e-20)
    df["dwt_noise_ratio"] = pd.Series(noise_ratio_raw, index=df.index).shift(1)

    # -----------------------------------------------------------------------
    # 10: DWT approximation ratio: approx_lN_energy / (total_energy + eps)
    # -----------------------------------------------------------------------
    approx_deepest = dwt_arrays[levels - 1]  # deepest approximation energy
    # Total energy = approx at deepest level + all detail energies
    total_e = approx_deepest.copy()
    for lv in range(levels):
        total_e = total_e + dwt_arrays[levels + lv]
    approx_ratio_raw = approx_deepest / (total_e + 1e-20)
    df["dwt_approx_ratio"] = pd.Series(approx_ratio_raw, index=df.index).shift(1)

    # -----------------------------------------------------------------------
    # 11-15: Spectral features (rolling FFT on returns)
    # -----------------------------------------------------------------------
    # Replace leading NaN with 0 for FFT window
    returns_clean = np.where(np.isnan(returns), 0.0, returns)
    spec_entropy, dom_freq, power_low, power_mid, power_high = _spectral_features_rolling(
        returns_clean, window=fft_window
    )
    df["spectral_entropy"] = pd.Series(spec_entropy, index=df.index).shift(1)
    df["dominant_freq"]    = pd.Series(dom_freq,     index=df.index).shift(1)
    df["power_low_freq"]   = pd.Series(power_low,    index=df.index).shift(1)
    df["power_mid_freq"]   = pd.Series(power_mid,    index=df.index).shift(1)
    df["power_high_freq"]  = pd.Series(power_high,   index=df.index).shift(1)

    # -----------------------------------------------------------------------
    # 16-20: Autocorrelation at lags 1, 2, 5, 10, 21
    # -----------------------------------------------------------------------
    returns_clean2 = np.where(np.isnan(returns), 0.0, returns)
    for lag in [1, 2, 5, 10, 21]:
        acorr = _rolling_autocorr(returns_clean2, lag=lag, window=autocorr_window)
        df[f"autocorr_lag{lag}"] = pd.Series(acorr, index=df.index).shift(1)

    # -----------------------------------------------------------------------
    # 21: Realized skewness (rolling 21d)
    # -----------------------------------------------------------------------
    returns_clean3 = np.where(np.isnan(returns), 0.0, returns)
    skew_arr = _rolling_skew(returns_clean3, window=moment_window)
    df["realized_skew"] = pd.Series(skew_arr, index=df.index).shift(1)

    # -----------------------------------------------------------------------
    # 22: Realized kurtosis (rolling 21d, excess)
    # -----------------------------------------------------------------------
    kurt_arr = _rolling_kurt(returns_clean3, window=moment_window)
    df["realized_kurt"] = pd.Series(kurt_arr, index=df.index).shift(1)

    # -----------------------------------------------------------------------
    # 23-24: Cross-correlation price returns vs volume changes at lag 0 and 1
    # -----------------------------------------------------------------------
    vol_chg_clean = np.where(np.isnan(vol_chg), 0.0, vol_chg)
    xcorr_l0 = _rolling_xcorr(returns_clean2, vol_chg_clean, lag=0, window=xcorr_window)
    xcorr_l1 = _rolling_xcorr(returns_clean2, vol_chg_clean, lag=1, window=xcorr_window)
    df["price_vol_xcorr_l0"] = pd.Series(xcorr_l0, index=df.index).shift(1)
    df["price_vol_xcorr_l1"] = pd.Series(xcorr_l1, index=df.index).shift(1)

    return df


# ---------------------------------------------------------------------------
# Exported column list
# ---------------------------------------------------------------------------

WAVELET_FEATURE_COLS: list[str] = [
    # DWT approximation energy — levels 1-4
    "dwt_approx_l1_energy",
    "dwt_approx_l2_energy",
    "dwt_approx_l3_energy",
    "dwt_approx_l4_energy",
    # DWT detail energy — levels 1-4
    "dwt_detail_l1_energy",
    "dwt_detail_l2_energy",
    "dwt_detail_l3_energy",
    "dwt_detail_l4_energy",
    # DWT derived ratios
    "dwt_noise_ratio",
    "dwt_approx_ratio",
    # Spectral features
    "spectral_entropy",
    "dominant_freq",
    "power_low_freq",
    "power_mid_freq",
    "power_high_freq",
    # Autocorrelation at multiple lags
    "autocorr_lag1",
    "autocorr_lag2",
    "autocorr_lag5",
    "autocorr_lag10",
    "autocorr_lag21",
    # Statistical moments
    "realized_skew",
    "realized_kurt",
    # Cross-correlation price returns vs volume
    "price_vol_xcorr_l0",
    "price_vol_xcorr_l1",
]
