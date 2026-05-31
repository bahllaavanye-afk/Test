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
    Roll (1984) spread estimator: 2 * sqrt(max(-Cov(Δp, Δp_lag), 0)).
    Approximates effective bid-ask spread from price changes.
    """
    dp = close.diff()
    dp_lag = dp.shift(1)
    cov = dp.rolling(window).cov(dp_lag)
    spread = 2 * (-cov).clip(lower=0).apply(np.sqrt)
    spread.name = "roll_spread"
    return spread


def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Corwin-Schultz (2012) high-low spread estimator.
    Uses 1-period and 2-period HL ratios to back out the spread.
    """
    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2
    # 2-period high/low
    high2 = high.rolling(2).max()
    low2 = low.rolling(2).min()
    log_hl2_sq = np.log(high2 / low2) ** 2

    # Rolling averages
    beta = log_hl_sq.rolling(window).mean() + log_hl_sq.shift(1).rolling(window).mean()
    gamma = log_hl2_sq.rolling(window).mean()

    alpha = (2 * (beta.apply(np.sqrt) - gamma.apply(np.sqrt))) / (3 - 2 * np.sqrt(2))
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0)
    spread.name = "cs_spread"
    return spread


def kyle_lambda_proxy(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Kyle's lambda (price impact) proxy: |return| / sqrt(volume).
    Higher lambda → larger price impact per unit volume.
    """
    impact = returns.abs() / (volume.apply(np.sqrt) + 1e-12)
    result = impact.rolling(window).mean()
    result.name = "kyle_lambda"
    return result


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------

def vwap_deviation(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Deviation of close from rolling VWAP: (close - VWAP) / VWAP."""
    typical = (close + high + low) / 3.0
    vwap = (typical * volume).rolling(window).sum() / (volume.rolling(window).sum() + 1e-12)
    result = (close - vwap) / (vwap + 1e-12)
    result.name = "vwap_dev"
    return result


def volume_price_trend(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Volume Price Trend (VPT) percent change over window."""
    vpt = (volume * close.pct_change()).cumsum()
    result = vpt.pct_change(window)
    result.name = "vpt_chg"
    return result


def on_balance_volume_momentum(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """OBV percent change over window — momentum of OBV."""
    direction = np.sign(close.diff()).fillna(0)
    obv_val = (direction * volume).cumsum()
    result = obv_val.pct_change(window)
    result.name = "obv_momentum"
    return result


def volume_weighted_rsi(
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Volume-weighted RSI: gains/losses weighted by volume before smoothing."""
    delta = close.diff()
    vol_norm = volume / (volume.rolling(window).mean() + 1e-12)

    gain = (delta.clip(lower=0) * vol_norm)
    loss = ((-delta).clip(lower=0) * vol_norm)

    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    result = 100 - (100 / (1 + rs))
    result.name = "vw_rsi"
    return result


# ---------------------------------------------------------------------------
# Advanced Momentum
# ---------------------------------------------------------------------------

def dema(close: pd.Series, span: int = 21) -> pd.Series:
    """Double EMA: 2*EMA(n) - EMA(EMA(n))."""
    ema1 = close.ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    result = 2 * ema1 - ema2
    result.name = f"dema_{span}"
    return result


def tema(close: pd.Series, span: int = 21) -> pd.Series:
    """Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    ema1 = close.ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    ema3 = ema2.ewm(span=span, adjust=False).mean()
    result = 3 * ema1 - 3 * ema2 + ema3
    result.name = f"tema_{span}"
    return result


def dema_diff(close: pd.Series, fast: int = 9, slow: int = 21) -> pd.Series:
    """Normalized DEMA crossover: (DEMA_fast - DEMA_slow) / close."""
    d_fast = dema(close, span=fast)
    d_slow = dema(close, span=slow)
    result = (d_fast - d_slow) / (close + 1e-12)
    result.name = "dema_diff"
    return result


def schaff_trend_cycle(
    close: pd.Series,
    fast: int = 23,
    slow: int = 50,
    k: int = 10,
) -> pd.Series:
    """Schaff Trend Cycle — stochastic of the MACD line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow

    # First stochastic
    macd_low = macd_line.rolling(k).min()
    macd_high = macd_line.rolling(k).max()
    stoch1 = (macd_line - macd_low) / (macd_high - macd_low + 1e-12) * 100

    # Smooth
    f1 = stoch1.ewm(span=3, adjust=False).mean()

    # Second stochastic on f1
    f1_low = f1.rolling(k).min()
    f1_high = f1.rolling(k).max()
    stoch2 = (f1 - f1_low) / (f1_high - f1_low + 1e-12) * 100

    result = stoch2.ewm(span=3, adjust=False).mean()
    result.name = "stc"
    return result


def kst_oscillator(close: pd.Series) -> pd.Series:
    """
    KST Oscillator — weighted sum of 4 smoothed ROC periods.
    KST = (RCMA1*1) + (RCMA2*2) + (RCMA3*3) + (RCMA4*4)
    where RCMA_i = SMA(ROC(r_i), s_i) with:
      roc periods: 10, 13, 14, 15
      sma periods: 10, 13, 14, 30
    """
    roc_periods = [10, 13, 14, 15]
    sma_periods = [10, 13, 14, 30]
    weights = [1, 2, 3, 4]

    kst = pd.Series(0.0, index=close.index)
    for rp, sp, w in zip(roc_periods, sma_periods, weights):
        roc = (close / close.shift(rp) - 1) * 100
        rcma = roc.rolling(sp).mean()
        kst = kst + w * rcma
    kst.name = "kst"
    return kst


def detrended_price_oscillator(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Detrended Price Oscillator: price.shift(window//2+1) - SMA(window).
    Removes longer-term trend to highlight cycles.
    """
    sma = close.rolling(window).mean()
    result = close.shift(window // 2 + 1) - sma
    result.name = "dpo"
    return result


def aroon_oscillator(
    high: pd.Series,
    low: pd.Series,
    window: int = 25,
) -> pd.Series:
    """
    Aroon Oscillator: (days_since_high - days_since_low) / window * 100.
    Range: [-100, +100]. Positive → uptrend.
    """
    def _days_since_max(arr):
        return len(arr) - 1 - np.argmax(arr)

    def _days_since_min(arr):
        return len(arr) - 1 - np.argmin(arr)

    days_high = high.rolling(window + 1).apply(_days_since_max, raw=True)
    days_low = low.rolling(window + 1).apply(_days_since_min, raw=True)
    result = (days_low - days_high) / window * 100
    result.name = "aroon_osc"
    return result


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Williams %R normalized to [-1, 0]: (close-highest_high)/(highest_high-lowest_low)."""
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    result = (close - hh) / (hh - ll + 1e-12)
    result.name = "williams_r"
    return result


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    s: int = 7,
    m: int = 14,
    l: int = 28,
) -> pd.Series:
    """
    Ultimate Oscillator: weighted combination of 3 time-period oscillators.
    Output range [0, 100], normalized to [-0.5, 0.5] for consistency.
    """
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - \
         pd.concat([low, prev_close], axis=1).min(axis=1)

    avg_s = bp.rolling(s).sum() / (tr.rolling(s).sum() + 1e-12)
    avg_m = bp.rolling(m).sum() / (tr.rolling(m).sum() + 1e-12)
    avg_l = bp.rolling(l).sum() / (tr.rolling(l).sum() + 1e-12)

    uo = 100 * (4 * avg_s + 2 * avg_m + avg_l) / 7
    # Normalize to [-0.5, 0.5]
    result = (uo / 100.0) - 0.5
    result.name = "ultimate_osc"
    return result


# ---------------------------------------------------------------------------
# Calendar / Cyclical Features (sin/cos encoded — no ordinal bias)
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical calendar encodings:
      dow_sin, dow_cos           — day of week (0-6)
      month_sin, month_cos       — month of year (1-12)
      quarter_end_proximity      — exp(-days_to_quarter_end/10)
      year_progress              — day_of_year/365
    """
    idx = pd.DatetimeIndex(df.index)

    dow = idx.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    month = idx.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Quarter end proximity: quarter ends at month 3, 6, 9, 12 on last day
    quarter_end_months = np.array([3, 6, 9, 12])
    days_to_qe = np.array([
        min(
            (pd.Timestamp(year=ts.year, month=qm, day=1) + pd.offsets.MonthEnd(0) - ts).days
            for qm in quarter_end_months
            if (pd.Timestamp(year=ts.year, month=qm, day=1) + pd.offsets.MonthEnd(0)) >= ts
        ) if any(
            (pd.Timestamp(year=ts.year, month=qm, day=1) + pd.offsets.MonthEnd(0)) >= ts
            for qm in quarter_end_months
        ) else 0
        for ts in idx
    ], dtype=float)
    df["quarter_end_proximity"] = np.exp(-days_to_qe / 10.0)

    doy = idx.day_of_year.astype(float)
    df["year_progress"] = doy / 365.0

    return df


# ---------------------------------------------------------------------------
# Regime Labels
# ---------------------------------------------------------------------------

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime classification columns:
      vol_regime      — 0=low, 1=normal, 2=high vs 252-bar rolling percentiles
      trend_regime    — +1/-1 vs SMA200
      momentum_regime — +1/-1 vs 21d return rolling 252-bar percentile
    """
    # Vol regime: use 21-day rolling std of returns as vol proxy
    returns = df["close"].pct_change()
    vol_21 = returns.rolling(21).std()

    low_thresh = vol_21.rolling(252, min_periods=50).quantile(0.33)
    high_thresh = vol_21.rolling(252, min_periods=50).quantile(0.67)

    vol_regime = pd.Series(1, index=df.index, dtype=int)  # default normal
    vol_regime[vol_21 <= low_thresh] = 0
    vol_regime[vol_21 >= high_thresh] = 2
    df["vol_regime"] = vol_regime.astype(float)

    # Trend regime: +1 if close > SMA200 else -1
    sma200 = df["close"].rolling(200, min_periods=50).mean()
    df["trend_regime"] = np.where(df["close"] > sma200, 1.0, -1.0)

    # Momentum regime: +1 if 21d return above 252-bar median else -1
    ret_21 = df["close"].pct_change(21)
    med_21 = ret_21.rolling(252, min_periods=50).median()
    df["momentum_regime"] = np.where(ret_21 > med_21, 1.0, -1.0)

    return df


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all advanced features and add them to df.
    Expects df to have columns: open, high, low, close, volume.
    Returns df with new feature columns appended.
    No lookahead: all features use only past data.
    """
    df = df.copy()

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]
    r = c.pct_change()

    # --- Volatility estimators ---
    df["gk_vol"] = garman_klass_vol(h, l, o, c, window=21)
    df["parkinson_vol"] = parkinson_vol(h, l, window=21)
    df["yz_vol"] = yang_zhang_vol(o, h, l, c, window=21)
    gk = df["gk_vol"]
    df["vol_pct_rank"] = vol_percentile_rank(gk, window=252)
    df["vol_of_vol"] = vol_of_vol(gk, window=21)

    # --- Complexity / Regime ---
    df["hurst_exponent"] = hurst_exponent(c, window=100)
    df["approx_entropy"] = approx_entropy(c, m=2, window=50)
    df["efficiency_ratio"] = efficiency_ratio(c, window=10)
    df["fractal_dim"] = fractal_dim_proxy(h, l, window=30)

    # --- Microstructure ---
    df["amihud_illiq"] = amihud_illiquidity(r, v, window=21)
    df["roll_spread"] = roll_spread(c, window=21)
    df["cs_spread"] = corwin_schultz_spread(h, l, window=21)
    df["kyle_lambda"] = kyle_lambda_proxy(r, v, window=21)

    # --- Volume profile ---
    df["vwap_dev"] = vwap_deviation(c, h, l, v, window=20)
    df["vpt_chg"] = volume_price_trend(c, v, window=20)
    df["obv_momentum"] = on_balance_volume_momentum(c, v, window=20)
    df["vw_rsi"] = volume_weighted_rsi(c, v, window=14)

    # --- Advanced momentum ---
    df["dema_diff"] = dema_diff(c, fast=9, slow=21)
    df["stc"] = schaff_trend_cycle(c, fast=23, slow=50, k=10)
    df["kst"] = kst_oscillator(c)
    df["dpo"] = detrended_price_oscillator(c, window=20)
    df["aroon_osc"] = aroon_oscillator(h, l, window=25)
    df["williams_r"] = williams_r(h, l, c, window=14)
    df["ultimate_osc"] = ultimate_oscillator(h, l, c, s=7, m=14, l=28)

    # --- Calendar features ---
    df = add_calendar_features(df)

    # --- Regime labels ---
    df = add_regime_features(df)

    return df


# Column list exported for feature engineering pipeline
ADVANCED_FEATURE_COLS: list[str] = [
    # Volatility estimators
    "gk_vol",
    "parkinson_vol",
    "yz_vol",
    "vol_pct_rank",
    "vol_of_vol",
    # Complexity / regime
    "hurst_exponent",
    "approx_entropy",
    "efficiency_ratio",
    "fractal_dim",
    # Microstructure
    "amihud_illiq",
    "roll_spread",
    "cs_spread",
    "kyle_lambda",
    # Volume profile
    "vwap_dev",
    "vpt_chg",
    "obv_momentum",
    "vw_rsi",
    # Advanced momentum
    "dema_diff",
    "stc",
    "kst",
    "dpo",
    "aroon_osc",
    "williams_r",
    "ultimate_osc",
    # Calendar cyclical
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "quarter_end_proximity",
    "year_progress",
    # Regime labels
    "vol_regime",
    "trend_regime",
    "momentum_regime",
]
