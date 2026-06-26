"""
Advanced indicator library — pure numpy/pandas, NO scipy.
All functions return pd.Series aligned to the input index.
No lookahead bias: all rolling windows look backward only.

Exports:
  add_advanced_features(df) -> pd.DataFrame
  ADVANCED_FEATURE_COLS: list[str]
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Volatility Estimators
# ---------------------------------------------------------------------------

def garman_klass_vol(
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Garman-Klass volatility estimator (annualized std proxy)."""
    try:
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        result = gk.rolling(window).mean().apply(lambda x: np.sqrt(max(x, 0)))
        result.name = "gk_vol"
        return result
    except (ZeroDivisionError, ValueError, TypeError) as e:
        logger.error(
            "Error in garman_klass_vol",
            extra={"function": "garman_klass_vol", "error": str(e)},
        )
        raise


def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Parkinson volatility estimator using high-low range."""
    try:
        log_hl_sq = np.log(high / low) ** 2
        factor = 1.0 / (4.0 * np.log(2))
        result = (log_hl_sq * factor).rolling(window).mean().apply(lambda x: np.sqrt(max(x, 0)))
        result.name = "parkinson_vol"
        return result
    except (ZeroDivisionError, ValueError, TypeError) as e:
        logger.error(
            "Error in parkinson_vol",
            extra={"function": "parkinson_vol", "error": str(e)},
        )
        raise


def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Yang-Zhang volatility estimator — robust to opening gaps."""
    try:
        log_oc = np.log(open_ / close.shift(1))      # overnight return
        log_co = np.log(close / open_)               # open-to-close return
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        sigma_oc = log_oc.rolling(window).var(ddof=1)
        sigma_co = log_co.rolling(window).var(ddof=1)
        rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()

        yz = sigma_oc + k * sigma_co + (1 - k) * rs
        result = yz.apply(lambda x: np.sqrt(max(x, 0)))
        result.name = "yang_zhang_vol"
        return result
    except (ZeroDivisionError, ValueError, TypeError) as e:
        logger.error(
            "Error in yang_zhang_vol",
            extra={"function": "yang_zhang_vol", "error": str(e)},
        )
        raise


def vol_percentile_rank(vol_series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling percentile rank of a volatility series, output in [0,1]."""
    def _pct_rank(arr):
        if len(arr) == 0:
            return np.nan
        return float(np.sum(arr[:-1] <= arr[-1])) / max(len(arr) - 1, 1)

    try:
        result = vol_series.rolling(window, min_periods=2).apply(_pct_rank, raw=True)
        result.name = "vol_pct_rank"
        return result
    except Exception as e:
        logger.error(
            "Error in vol_percentile_rank",
            extra={"function": "vol_percentile_rank", "error": str(e)},
        )
        raise


def vol_of_vol(vol_series: pd.Series, window: int = 21) -> pd.Series:
    """Standard deviation of a volatility series (vol-of-vol)."""
    try:
        result = vol_series.rolling(window).std(ddof=1)
        result.name = "vol_of_vol"
        return result
    except Exception as e:
        logger.error(
            "Error in vol_of_vol",
            extra={"function": "vol_of_vol", "error": str(e)},
        )
        raise


# ---------------------------------------------------------------------------
# Complexity / Regime
# ---------------------------------------------------------------------------

def hurst_exponent(prices: pd.Series, window: int = 100) -> pd.Series:
    """
    Hurst exponent via R/S analysis — pure numpy, no scipy.
    H ≈ 0.5 → random walk; H > 0.5 → trending; H < 0.5 → mean-reverting.
    """
    def _hurst(arr):
        n = len(arr)
        if n < 20:
            return 0.5
        lags = [max(2, n // 8), max(4, n // 4), max(8, n // 2), max(16, n * 3 // 4)]
        lags = sorted(set(l for l in lags if 2 <= l < n))
        if len(lags) < 2:
            return 0.5
        rs_vals = []
        for lag in lags:
            sub = arr[:lag]
            mean_sub = np.mean(sub)
            deviations = np.cumsum(sub - mean_sub)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(sub, ddof=1)
            if s < 1e-12:
                rs_vals.append(np.nan)
            else:
                rs_vals.append(r / s)
        rs_arr = np.array(rs_vals, dtype=float)
        lags_arr = np.array(lags, dtype=float)
        valid = ~np.isnan(rs_arr) & (rs_arr > 0) & (lags_arr > 0)
        if valid.sum() < 2:
            return 0.5
        log_rs = np.log(rs_arr[valid])
        log_lags = np.log(lags_arr[valid])
        # OLS slope
        x = log_lags - log_lags.mean()
        y = log_rs - log_rs.mean()
        denom = np.dot(x, x)
        if denom < 1e-12:
            return 0.5
        return float(np.dot(x, y) / denom)

    try:
        result = prices.rolling(window, min_periods=20).apply(_hurst, raw=True)
        result.name = "hurst_exponent"
        return result
    except Exception as e:
        logger.error(
            "Error in hurst_exponent",
            extra={"function": "hurst_exponent", "error": str(e)},
        )
        raise


def approx_entropy(series: pd.Series, m: int = 2, window: int = 50) -> pd.Series:
    """
    Rolling Approximate Entropy — pure numpy.
    Lower ApEn → more regular/predictable; higher → more complex/random.
    """
    def _apen(arr):
        n = len(arr)
        if n < m + 2:
            return np.nan
        r = 0.2 * np.std(arr, ddof=1)
        if r < 1e-12:
            return 0.0

        def _phi(m_):
            count = 0
            total = 0
            for i in range(n - m_):
                template = arr[i:i + m_]
                for j in range(n - m_):
                    if np.max(np.abs(arr[j:j + m_] - template)) <= r:
                        count += 1
                total += 1
            if total == 0 or count == 0:
                return 0.0
            return np.log(count / total)

        return float(_phi(m) - _phi(m + 1))

    try:
        result = series.rolling(window, min_periods=m + 2).apply(_apen, raw=True)
        result.name = "approx_entropy"
        return result
    except Exception as e:
        logger.error(
            "Error in approx_entropy",
            extra={"function": "approx_entropy", "error": str(e)},
        )
        raise


def efficiency_ratio(prices: pd.Series, window: int = 10) -> pd.Series:
    """
    Kaufman Efficiency Ratio: |net change| / sum(|bar changes|), in [0,1].
    ER→1 = trending cleanly; ER→0 = choppy/random.
    """
    try:
        net_change = prices.diff(window).abs()
        path_length = prices.diff().abs().rolling(window).sum()
        result = net_change / (path_length + 1e-12)
        result = result.clip(0, 1)
        result.name = "efficiency_ratio"
        return result
    except Exception as e:
        logger.error(
            "Error in efficiency_ratio",
            extra={"function": "efficiency_ratio", "error": str(e)},
        )
        raise


def fractal_dim_proxy(
    high: pd.Series,
    low: pd.Series,
    window: int = 30,
) -> pd.Series:
    """
    Fractal dimension proxy using the HL range ratio method — pure numpy.
    Values near 1 → trending; near 2 → random/choppy.
    """
    def _fd(arr_h, arr_l):
        n = len(arr_h)
        if n < 4:
            return 1.5
        half = n // 2
        # Range of first half, second half, full period
        r1 = np.max(arr_h[:half]) - np.min(arr_l[:half])
        r2 = np.max(arr_h[half:]) - np.min(arr_l[half:])
        r_full = np.max(arr_h) - np.min(arr_l)
        if r_full < 1e-12:
            return 1.5
        # FD = log(r1+r2) / log(r_full * 2) approximately
        denom = np.log(r_full) + np.log(2)
        numer = np.log(r1 + r2 + 1e-12)
        if abs(denom) < 1e-12:
            return 1.5
        return float(numer / denom)

    try:
        # Rolling apply on aligned high/low arrays
        highs = high.values
        lows = low.values
        n = len(highs)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            out[i] = _fd(highs[i - window + 1:i + 1], lows[i - window + 1:i + 1])

        result = pd.Series(out, index=high.index, name="fractal_dim")
        return result
    except Exception as e:
        logger.error(
            "Error in fractal_dim_proxy",
            extra={"function": "fractal_dim_proxy", "error": str(e)},
        )
        raise


# ---------------------------------------------------------------------------
# Microstructure (OHLCV-based)
# ---------------------------------------------------------------------------

def amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Amihud illiquidity ratio: |r| / (|r| * close * volume) proxy × 1e6.
    Uses |return| / dollar_volume * 1e6 (approximation without price).
    """
    try:
        dollar_vol = volume.abs() + 1e-12  # approximate dollar vol without price
        illiq = (returns.abs() / dollar_vol) * 1e6
        result = illiq.rolling(window).mean()
        result.name = "amihud_illiq"
        return result
    except Exception as e:
        logger.error(
            "Error in amihud_illiquidity",
            extra={"function": "amihud_illiquidity", "error": str(e)},
        )
        raise


def roll_spread(close: pd.Series, window: int = 21) -> pd.Series:
    """
    Roll (1984) spread estimator: 2 * sq
# ... (truncated for brevity)
    """
    try:
        # Placeholder implementation – actual logic should be added.
        result = pd.Series(np.nan, index=close.index, name="roll_spread")
        return result
    except Exception as e:
        logger.error(
            "Error in roll_spread",
            extra={"function": "roll_spread", "error": str(e)},
        )
        raise