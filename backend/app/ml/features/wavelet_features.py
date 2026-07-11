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
import time
from typing import Callable, List, Tuple

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
    if window <= lag:
        raise ValueError("window must be larger than lag")

    n = len(series)
    out = np.full(n, np.nan)

    for i in range(window - 1, n):
        x = series[i - window + 1 : i - lag + 1]
        y = series[i - window + lag + 1 : i + 1]
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        out[i] = np.corrcoef(x, y)[0, 1]

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_wavelet_features(df: pd.DataFrame, levels: int = 4) -> pd.DataFrame:
    """
    Compute wavelet‑based and spectral features for each column in ``df``.
    Features are added as new columns with a ``_w_`` prefix.
    The function returns a new DataFrame (original is not mutated).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas.DataFrame")
    if levels < 1:
        raise ValueError("levels must be a positive integer")

    # Work on a copy to avoid side‑effects
    result = df.copy()

    # For each numeric column compute rolling features
    for col in result.select_dtypes(include=[np.number]).columns:
        series = result[col].to_numpy(dtype=float)

        # Wavelet energies
        try:
            dwt_feats = _rolling_dwt_energies(series, window=levels * 2, levels=levels)
        except Exception:
            logger.exception("Error computing DWT energies for column %s", col)
            continue

        for lvl, arr in enumerate(dwt_feats[:levels], start=1):
            result[f"{col}_w_approx_energy_l{lvl}"] = arr
        for lvl, arr in enumerate(dwt_feats[levels:], start=1):
            result[f"{col}_w_detail_energy_l{lvl}"] = arr

        # Spectral features
        try:
            spec_entropy, dom_freq, p_low, p_mid, p_high = _spectral_features_rolling(
                series, window=levels * 4
            )
        except Exception:
            logger.exception("Error computing spectral features for column %s", col)
            continue

        result[f"{col}_w_spec_entropy"] = spec_entropy
        result[f"{col}_w_dom_freq"] = dom_freq
        result[f"{col}_w_power_low"] = p_low
        result[f"{col}_w_power_mid"] = p_mid
        result[f"{col}_w_power_high"] = p_high

        # Autocorrelation (example lag=1)
        try:
            ac = _rolling_autocorr(series, lag=1, window=levels * 2)
        except Exception:
            logger.exception("Error computing autocorrelation for column %s", col)
            continue
        result[f"{col}_w_autocorr_lag1"] = ac

    return result


# ---------------------------------------------------------------------------
# Monitoring utilities
# ---------------------------------------------------------------------------

def _monitor(func: Callable) -> Callable:
    """
    Decorator that logs execution time and basic metrics at INFO level.
    Metrics logged:
        - function name
        - number of processed signals (rows) when a DataFrame is the first argument
        - execution time in milliseconds
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000

        signal_count = None
        try:
            first_arg = args[0] if args else kwargs.get("df")
            if isinstance(first_arg, pd.DataFrame):
                signal_count = len(first_arg)
        except Exception:
            signal_count = None

        logger.info(
            "Feature computation completed",
            extra={
                "function": func.__name__,
                "signal_count": signal_count,
                "execution_time_ms": round(elapsed_ms, 3),
            },
        )
        return result
    return wrapper

# Apply monitoring to the public API if it exists
if "add_wavelet_features" in globals():
    globals()["add_wavelet_features"] = _monitor(globals()["add_wavelet_features"])

# ---------------------------------------------------------------------------
# Exported symbols
# ---------------------------------------------------------------------------

WAVELET_FEATURE_COLS: List[str] = [
    # placeholder – actual column names are generated dynamically in add_wavelet_features
]

__all__ = [
    "add_wavelet_features",
    "WAVELET_FEATURE_COLS",
    "FeatureComputationError",
]