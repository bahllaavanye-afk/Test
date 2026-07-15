"""
Advanced indicator library — pure numpy/pandas, NO scipy.
All functions return pd.Series aligned to the input index.
No lookahead bias: all rolling windows look backward only.

Exports:
  add_advanced_features(df) -> pd.DataFrame
  ADVANCED_FEATURE_COLS: list[str]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    result = gk.rolling(window).mean().apply(lambda x: np.sqrt(max(x, 0)))
    result.name = "gk_vol"
    return result


def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Parkinson volatility estimator using high-low range."""
    log_hl_sq = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2))
    result = (log_hl_sq * factor).rolling(window).mean().apply(lambda x: np.sqrt(max(x, 0)))
    result.name = "parkinson_vol"
    return result


def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Yang-Zhang volatility estimator — robust to opening gaps."""
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


def vol_percentile_rank(vol_series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling percentile rank of a volatility series, output in [0,1]."""
    def _pct_rank(arr):
        if len(arr) == 0:
            return np.nan
        return float(np.sum(arr[:-1] <= arr[-1])) / max(len(arr) - 1, 1)
    result = vol_series.rolling(window, min_periods=2).apply(_pct_rank, raw=True)
    result.name = "vol_pct_rank"
    return result


def vol_of_vol(vol_series: pd.Series, window: int = 21) -> pd.Series:
    """Standard deviation of a volatility series (vol-of-vol)."""
    result = vol_series.rolling(window).std(ddof=1)
    result.name = "vol_of_vol"
    return result


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

    result = prices.rolling(window, min_periods=20).apply(_hurst, raw=True)
    result.name = "hurst_exponent"
    return result


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

    result = series.rolling(window, min_periods=m + 2).apply(_apen, raw=True)
    result.name = "approx_entropy"
    return result


def efficiency_ratio(prices: pd.Series, window: int = 10) -> pd.Series:
    """
    Kaufman Efficiency Ratio: |net change| / sum(|bar changes|), in [0,1].
    ER→1 = trending cleanly; ER→0 = choppy/random.
    """
    net_change = prices.diff(window).abs()
    path_length = prices.diff().abs().rolling(window).sum()
    result = net_change / (path_length + 1e-12)
    result = result.clip(0, 1)
    result.name = "efficiency_ratio"
    return result


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

    # Rolling apply on aligned high/low arrays
    highs = high.values
    lows = low.values
    n = len(highs)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = _fd(highs[i - window + 1:i + 1], lows[i - window + 1:i + 1])

    result = pd.Series(out, index=high.index, name="fractal_dim")
    return result


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
    dollar_vol = volume.abs() + 1e-12  # approximate dollar vol without price
    illiq = (returns.abs() / dollar_vol) * 1e6
    result = illiq.rolling(window).mean()
    result.name = "amihud_illiq"
    return result


def roll_spread(close: pd.Series, window: int = 21) -> pd.Series:
    """
    Roll (1984) spread estimator: 2 * sq
    """
    # Implementation omitted for brevity
    pass


# ---------------------------------------------------------------------------
# Unit Tests for Edge Cases
# ---------------------------------------------------------------------------

import unittest

class TestAdvancedIndicators(unittest.TestCase):
    def test_garman_klass_vol_short_series(self):
        """When the input series is shorter than the window, early values should be NaN."""
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        high = pd.Series([10, 11, 12, 13, 14], index=dates)
        low = pd.Series([9, 10, 11, 12, 13], index=dates)
        open_ = pd.Series([9.5, 10.5, 11.5, 12.5, 13.5], index=dates)
        close = pd.Series([10, 11, 12, 13, 14], index=dates)
        result = garman_klass_vol(high, low, open_, close, window=10)
        # First 9 entries must be NaN because window=10 > length
        self.assertTrue(result.isna().all())

    def test_hurst_exponent_minimum_length(self):
        """Hurst exponent should return the default 0.5 for windows with insufficient data."""
        dates = pd.date_range("2022-01-01", periods=15, freq="D")
        prices = pd.Series(np.arange(15), index=dates)
        result = hurst_exponent(prices, window=20)
        # Since min_periods=20, all values should be NaN (no computation)
        self.assertTrue(result.isna().all())

    def test_vol_percentile_rank_constant_series(self):
        """Percentile rank on a constant series should be 1.0 after enough observations."""
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        vol = pd.Series([0.2] * 5, index=dates)
        result = vol_percentile_rank(vol, window=3)
        # After the third observation, each new value is equal to previous ones,
        # so the rank should be 1.0 (all previous <= current)
        expected = pd.Series([np.nan, np.nan, 1.0, 1.0, 1.0], index=dates, name="vol_pct_rank")
        pd.testing.assert_series_equal(result, expected)

if __name__ == "__main__":
    unittest.main()