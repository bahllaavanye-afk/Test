"""Bot evaluation engine — checks triggers, conditions, and executes actions."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, time as dtime
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.bot import Bot
from app.schemas.bot import ConditionConfig, ActionConfig, ExitRuleConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Return the most recent RSI value."""
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def compute_sma(prices: pd.Series, period: int = 50) -> float:
    """Return the most recent SMA value."""
    if len(prices) < period:
        return float(prices.mean())
    return float(prices.iloc[-period:].mean())


def compute_ema(prices: pd.Series, period: int = 20) -> float:
    """Return the most recent EMA value."""
    if len(prices) < period:
        return float(prices.mean())
    return float(prices.ewm(span=period, adjust=False).mean().iloc[-1])


def compute_macd(prices: pd.Series) -> tuple[float, float]:
    """Return (macd_line, signal_line) most recent values."""
    if len(prices) < 26:
        return 0.0, 0.0
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def compute_bb(prices: pd.Series, period: int = 20) -> tuple[float, float, float]:
    """Return (upper, middle, lower) Bollinger Bands."""
    if len(prices) < period:
        mean = float(prices.mean())
        return mean, mean, mean
    window = prices.iloc[-period:]
    mid = float(window.mean())
    std = float(window.std())
    return mid + 2 * std, mid, mid - 2 * std


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Return the most recent Average True Range value."""
    if len(close) < period + 1:
        return 0.0
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return float(atr.iloc[-1])


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[float, float, float]:
    """Return (adx, plus_di, minus_di) most recent values."""
    if len(close) < period * 2 + 1:
        return 0.0, 0.0, 0.0
    try:
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high - prev_high
        minus_dm = prev_low - low
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # Smooth
        atr_s = tr.ewm(com=period - 1, min_periods=period).mean()
        plus_dm_s = plus_dm.ewm(com=period - 1, min_periods=period).mean()
        minus_dm_s = minus_dm.ewm(com=period - 1, min_periods=period).mean()

        plus_di = 100.0 * plus_dm_s / atr_s.replace(0, np.nan)
        minus_di = 100.0 * minus_dm_s / atr_s.replace(0, np.nan)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(com=period - 1, min_periods=period).mean()

        return (
            float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0,
            float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0.0,
            float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0.0,
        )
    except Exception:
        return 0.0, 0.0, 0.0


def compute_stoch_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[float, float]:
    """Return (K, D) StochRSI values (0-100 scale)."""
    if len(close) < rsi_period + stoch_period + k_period + d_period:
        return 50.0, 50.0
    try:
        # Build full RSI series
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_series = rsi_series.fillna(50.0)

        # Stochastic of RSI
        rsi_min = rsi_series.rolling(stoch_period).min()
        rsi_max = rsi_series.rolling(stoch_period).max()
        stoch = 100.0 * (rsi_series - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
        stoch = stoch.fillna(50.0)

        k_line = stoch.rolling(k_period).mean().fillna(50.0)
        d_line = k_line.rolling(d_period).mean().fillna(50.0)

        return float(k_line.iloc[-1]), float(d_line.iloc[-1])
    except Exception:
        return 50.0, 50.0


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3,
) -> tuple[float, float]:
    """Return (K, D) Stochastic Oscillator values (0-100 scale)."""
    if len(close) < k_period + d_period:
        return 50.0, 50.0
    try:
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        k = k.fillna(50.0)
        d = k.rolling(d_period).mean().fillna(50.0)
        return float(k.iloc[-1]), float(d.iloc[-1])
    except Exception:
        return 50.0, 50.0


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> float:
    """Return the most recent Commodity Channel Index value."""
    if len(close) < period:
        return 0.0
    try:
        typical_price = (high + low + close) / 3.0
        tp_window = typical_price.iloc[-period:]
        mean_tp = float(tp_window.mean())
        mean_dev = float((tp_window - mean_tp).abs().mean())
        if mean_dev == 0:
            return 0.0
        return float((typical_price.iloc[-1] - mean_tp) / (0.015 * mean_dev))
    except Exception:
        return 0.0


def compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float:
    """Return the most recent Williams %R value (-100 to 0)."""
    if len(close) < period:
        return -50.0
    try:
        highest_high = float(high.iloc[-period:].max())
        lowest_low = float(low.iloc[-period:].min())
        if highest_high == lowest_low:
            return -50.0
        return float(-100.0 * (highest_high - close.iloc[-1]) / (highest_high - lowest_low))
    except Exception:
        return -50.0


def compute_obv(close: pd.Series, volume: pd.Series) -> float:
    """Return the most recent On-Balance Volume value."""
    if len(close) < 2:
        return 0.0
    try:
        direction = np.sign(close.diff().fillna(0))
        obv = (direction * volume).cumsum()
        return float(obv.iloc[-1])
    except Exception:
        return 0.0


def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> float:
    """Return the session VWAP (cumulative from start of series)."""
    if len(close) < 1:
        return float(close.iloc[-1]) if len(close) > 0 else 0.0
    try:
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = (typical_price * volume).cumsum()
        cum_vol = volume.cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        return float(vwap.iloc[-1])
    except Exception:
        return float(close.iloc[-1]) if len(close) > 0 else 0.0


def compute_supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0,
) -> tuple[float, bool]:
    """Return (supertrend_value, is_uptrend)."""
    if len(close) < period + 1:
        return float(close.iloc[-1]) if len(close) > 0 else 0.0, True
    try:
        atr = compute_atr(high, low, close, period)
        hl2 = (high + low) / 2.0

        # Basic upper/lower bands
        upper_basic = hl2 + multiplier * atr
        lower_basic = hl2 - multiplier * atr

        # Build supertrend iteratively using last N+1 points
        n = len(close)
        final_upper = [0.0] * n
        final_lower = [0.0] * n
        supertrend = [0.0] * n
        is_up = [True] * n

        for i in range(1, n):
            fu_prev = final_upper[i - 1]
            fl_prev = final_lower[i - 1]

            ub = float(upper_basic.iloc[i])
            lb = float(lower_basic.iloc[i])

            final_upper[i] = ub if (ub < fu_prev or float(close.iloc[i - 1]) > fu_prev) else fu_prev
            final_lower[i] = lb if (lb > fl_prev or float(close.iloc[i - 1]) < fl_prev) else fl_prev

            cl = float(close.iloc[i])
            if supertrend[i - 1] == final_upper[i - 1]:
                is_up[i] = cl > final_upper[i]
            else:
                is_up[i] = cl >= final_lower[i]

            supertrend[i] = final_lower[i] if is_up[i] else final_upper[i]

        return float(supertrend[-1]), bool(is_up[-1])
    except Exception:
        return float(close.iloc[-1]) if len(close) > 0 else 0.0, True


def compute_ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, float]:
    """Return Ichimoku Cloud components: tenkan, kijun, senkou_a, senkou_b, chikou."""
    safe = {
        "tenkan": float(close.iloc[-1]) if len(close) > 0 else 0.0,
        "kijun": float(close.iloc[-1]) if len(close) > 0 else 0.0,
        "senkou_a": float(close.iloc[-1]) if len(close) > 0 else 0.0,
        "senkou_b": float(close.iloc[-1]) if len(close) > 0 else 0.0,
        "chikou": float(close.iloc[-1]) if len(close) > 0 else 0.0,
    }
    if len(close) < 52:
        return safe
    try:
        def midpoint(s_high: pd.Series, s_low: pd.Series, n: int) -> float:
            return float((s_high.iloc[-n:].max() + s_low.iloc[-n:].min()) / 2.0)

        tenkan = midpoint(high, low, 9)
        kijun = midpoint(high, low, 26)
        senkou_a = (tenkan + kijun) / 2.0
        senkou_b = midpoint(high, low, 52)
        chikou = float(close.iloc[-1])

        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou": chikou,
        }
    except Exception:
        return safe


def compute_pivot_points(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, float]:
    """Return classic pivot points: pp, r1, r2, r3, s1, s2, s3 (based on last complete session)."""
    safe_price = float(close.iloc[-1]) if len(close) > 0 else 0.0
    safe = {"pp": safe_price, "r1": safe_price, "r2": safe_price, "r3": safe_price,
            "s1": safe_price, "s2": safe_price, "s3": safe_price}
    if len(close) < 2:
        return safe
    try:
        # Use previous session values
        h = float(high.iloc[-2])
        l = float(low.iloc[-2])
        c = float(close.iloc[-2])

        pp = (h + l + c) / 3.0
        r1 = 2.0 * pp - l
        s1 = 2.0 * pp - h
        r2 = pp + (h - l)
        s2 = pp - (h - l)
        r3 = h + 2.0 * (pp - l)
        s3 = l - 2.0 * (h - pp)

        return {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}
    except Exception:
        return safe


def compute_momentum(close: pd.Series, period: int = 10) -> float:
    """Return Rate of Change % over the given period."""
    if len(close) < period + 1:
        return 0.0
    try:
        past = float(close.iloc[-period - 1])
        if past == 0:
            return 0.0
        return float((close.iloc[-1] - past) / past * 100.0)
    except Exception:
        return 0.0


def compute_mfi(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, period: int = 14,
) -> float:
    """Return the most recent Money Flow Index value."""
    if len(close) < period + 1:
        return 50.0
    try:
        typical_price = (high + low + close) / 3.0
        raw_mf = typical_price * volume

        tp_diff = typical_price.diff()
        pos_mf = raw_mf.where(tp_diff > 0, 0.0)
        neg_mf = raw_mf.where(tp_diff < 0, 0.0)

        pos_sum = pos_mf.rolling(period).sum().iloc[-1]
        neg_sum = neg_mf.rolling(period).sum().iloc[-1]

        if neg_sum == 0:
            return 100.0
        mfr = pos_sum / neg_sum
        return float(100.0 - (100.0 / (1.0 + mfr)))
    except Exception:
        return 50.0


def compute_ppo(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float]:
    """Return (ppo_line, signal_line) most recent values (as percentages)."""
    if len(close) < slow + signal:
        return 0.0, 0.0
    try:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        ppo_line = 100.0 * (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
        ppo_line = ppo_line.fillna(0.0)
        signal_line = ppo_line.ewm(span=signal, adjust=False).mean()
        return float(ppo_line.iloc[-1]), float(signal_line.iloc[-1])
    except Exception:
        return 0.0, 0.0


def _compare(a: float, op: str, b: float) -> bool:
    if op == "<":
        return a < b
    if op == ">":
        return a > b
    if op == "<=":
        return a <= b
    if op == ">=":
        return a >= b
    if op == "==":
        return abs(a - b) < 1e-9
    if op == "!=":
        return abs(a - b) >= 1e-9
    return False


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------

def evaluate_condition(cond: ConditionConfig, data: pd.DataFrame, current_price: float) -> bool:  # noqa: C901
    """Evaluate a single condition against the data and current price."""
    close = data["close"] if "close" in data.columns else pd.Series([current_price])

    # Extract OHLCV columns with safe fallbacks
    high = data["high"] if "high" in data.columns else close
    low = data["low"] if "low" in data.columns else close
    volume = data["volume"] if "volume" in data.columns else pd.Series([0.0] * len(close))

    ctype = cond.type

    if ctype == "indicator":
        ind = (cond.indicator or "").lower()

        # ── existing indicators ─────────────────────────────────────────────

        if ind == "rsi":
            val = compute_rsi(close, cond.period)

        elif ind == "sma":
            val = compute_sma(close, cond.period)

        elif ind == "ema":
            val = compute_ema(close, cond.period)

        elif ind == "bb":
            upper, mid, lower = compute_bb(close, cond.period)
            op = cond.operator
            if op == "price_below_lower":
                return current_price < lower
            if op == "price_above_upper":
                return current_price > upper
            if op == "price_at_middle":
                # within 1% of middle band
                return abs(current_price - mid) / max(mid, 1e-9) < 0.01
            if op == "bb_squeeze":
                band_width = (upper - lower) / max(mid, 1e-9)
                return band_width < 0.02
            return False

        elif ind == "macd":
            macd_val, signal_val = compute_macd(close)
            op = cond.operator
            if op == "bullish_cross":
                if len(close) < 28:
                    return False
                macd_prev, sig_prev = compute_macd(close.iloc[:-1])
                return (macd_prev <= sig_prev) and (macd_val > signal_val)
            if op == "bearish_cross":
                if len(close) < 28:
                    return False
                macd_prev, sig_prev = compute_macd(close.iloc[:-1])
                return (macd_prev >= sig_prev) and (macd_val < signal_val)
            val = macd_val

        elif ind == "ema_cross":
            op = cond.operator
            fast_p = cond.fast_period or 20
            slow_p = cond.slow_period or 50
            min_len = slow_p + 1
            if len(close) < min_len:
                return False
            ema_fast_now = compute_ema(close, fast_p)
            ema_slow_now = compute_ema(close, slow_p)
            ema_fast_prev = compute_ema(close.iloc[:-1], fast_p)
            ema_slow_prev = compute_ema(close.iloc[:-1], slow_p)
            if op == "bullish_cross":
                return (ema_fast_prev <= ema_slow_prev) and (ema_fast_now > ema_slow_now)
            if op == "bearish_cross":
                return (ema_fast_prev >= ema_slow_prev) and (ema_fast_now < ema_slow_now)
            return False

        # ── new indicators ──────────────────────────────────────────────────

        elif ind == "atr":
            val = compute_atr(high, low, close, cond.period)

        elif ind == "adx":
            adx_val, plus_di, minus_di = compute_adx(high, low, close, cond.period)
            op = cond.operator
            if op == "adx_above":
                if cond.value is None:
                    return False
                return adx_val > cond.value
            if op == "trend_strong":
                return adx_val > 25.0
            # numeric compare on ADX
            val = adx_val

        elif ind == "stoch":
            k_p = cond.k_period or 14
            d_p = cond.d_period or 3
            k_val, d_val = compute_stochastic(high, low, close, k_p, d_p)
            op = cond.operator
            if op == "overbought":
                return k_val > 80.0
            if op == "oversold":
                return k_val < 20.0
            val = k_val

        elif ind == "stoch_rsi":
            k_p = cond.k_period or 3
            d_p = cond.d_period or 3
            k_val, d_val = compute_stoch_rsi(close, cond.period, cond.period, k_p, d_p)
            op = cond.operator
            if op == "overbought":
                return k_val > 80.0
            if op == "oversold":
                return k_val < 20.0
            val = k_val

        elif ind == "cci":
            val = compute_cci(high, low, close, cond.period)
            op = cond.operator
            if op == "overbought":
                return val > 100.0
            if op == "oversold":
                return val < -100.0

        elif ind == "williams_r":
            val = compute_williams_r(high, low, close, cond.period)
            op = cond.operator
            if op == "overbought":
                return val > -20.0
            if op == "oversold":
                return val < -80.0

        elif ind == "obv":
            val = compute_obv(close, volume)

        elif ind == "vwap":
            vwap_val = compute_vwap(high, low, close, volume)
            op = cond.operator
            if op == "price_above_vwap":
                return current_price > vwap_val
            if op == "price_below_vwap":
                return current_price < vwap_val
            val = vwap_val

        elif ind == "supertrend":
            mult = cond.multiplier if cond.multiplier is not None else 3.0
            st_val, is_uptrend = compute_supertrend(high, low, close, cond.period, mult)
            op = cond.operator
            if op == "bullish":
                return is_uptrend
            if op == "bearish":
                return not is_uptrend
            val = st_val

        elif ind == "ichimoku":
            ichi = compute_ichimoku(high, low, close)
            op = cond.operator
            cloud_top = max(ichi["senkou_a"], ichi["senkou_b"])
            cloud_bot = min(ichi["senkou_a"], ichi["senkou_b"])
            if op == "price_above_cloud":
                return current_price > cloud_top
            if op == "price_below_cloud":
                return current_price < cloud_bot
            if op == "tenkan_kijun_cross_bull":
                # tenkan crosses above kijun — need previous bar
                if len(close) < 53:
                    return False
                ichi_prev = compute_ichimoku(high.iloc[:-1], low.iloc[:-1], close.iloc[:-1])
                return (ichi_prev["tenkan"] <= ichi_prev["kijun"]) and (ichi["tenkan"] > ichi["kijun"])
            if op == "tenkan_kijun_cross_bear":
                if len(close) < 53:
                    return False
                ichi_prev = compute_ichimoku(high.iloc[:-1], low.iloc[:-1], close.iloc[:-1])
                return (ichi_prev["tenkan"] >= ichi_prev["kijun"]) and (ichi["tenkan"] < ichi["kijun"])
            # numeric: compare tenkan to value
            val = ichi["tenkan"]

        elif ind == "mfi":
            val = compute_mfi(high, low, close, volume, cond.period)
            op = cond.operator
            if op == "overbought":
                return val > 80.0
            if op == "oversold":
                return val < 20.0

        elif ind == "momentum":
            val = compute_momentum(close, cond.period)

        elif ind == "ppo":
            ppo_val, sig_val = compute_ppo(close)
            op = cond.operator
            if op == "bullish_cross":
                if len(close) < 36:
                    return False
                ppo_prev, sig_prev = compute_ppo(close.iloc[:-1])
                return (ppo_prev <= sig_prev) and (ppo_val > sig_val)
            if op == "bearish_cross":
                if len(close) < 36:
                    return False
                ppo_prev, sig_prev = compute_ppo(close.iloc[:-1])
                return (ppo_prev >= sig_prev) and (ppo_val < sig_val)
            val = ppo_val

        elif ind == "pivot_support":
            pivots = compute_pivot_points(high, low, close)
            supports = sorted([pivots["s1"], pivots["s2"], pivots["s3"]], reverse=True)
            # Find nearest support below current price
            nearest = next((s for s in supports if s <= current_price), supports[-1])
            op = cond.operator
            if cond.value is not None:
                # compare distance to nearest support (in %)
                distance_pct = abs(current_price - nearest) / max(current_price, 1e-9) * 100.0
                return _compare(distance_pct, op, cond.value)
            return current_price > nearest

        elif ind == "pivot_resistance":
            pivots = compute_pivot_points(high, low, close)
            resistances = sorted([pivots["r1"], pivots["r2"], pivots["r3"]])
            # Find nearest resistance above current price
            nearest = next((r for r in resistances if r >= current_price), resistances[-1])
            op = cond.operator
            if cond.value is not None:
                distance_pct = abs(nearest - current_price) / max(current_price, 1e-9) * 100.0
                return _compare(distance_pct, op, cond.value)
            return current_price < nearest

        else:
            logger.warning("Unknown indicator", indicator=ind)
            return False

        # Numeric compare (fall-through for indicators that set `val`)
        if cond.value is None:
            return False
        return _compare(val, cond.operator, cond.value)

    elif ctype == "price_vs_ma":
        period = cond.ma_period or 50
        ma_type = (cond.ma_type or "sma").lower()
        if ma_type == "ema":
            ma = compute_ema(close, period)
        else:
            ma = compute_sma(close, period)
        return _compare(current_price, cond.operator, ma)

    elif ctype == "pnl":
        if len(close) < 2:
            return False
        day_pnl_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
        threshold = cond.pnl_pct if cond.pnl_pct is not None else 0.0
        return _compare(day_pnl_pct, cond.operator, threshold)

    elif ctype == "time_window":
        from datetime import timedelta
        now_et = (datetime.now(timezone.utc) - timedelta(hours=5)).time()
        try:
            start = dtime(*[int(x) for x in (cond.start_time or "09:30").split(":")])
            end = dtime(*[int(x) for x in (cond.end_time or "16:00").split(":")])
            return start <= now_et <= end
        except Exception:
            return True

    elif ctype == "position_exists":
        return True

    elif ctype == "no_position":
        return True

    return False


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _map_crypto_symbol(symbol: str) -> str:
    """Map exchange crypto symbols to yfinance format."""
    mapping = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "SOLUSDT": "SOL-USD",
        "BNBUSDT": "BNB-USD",
        "ADAUSDT": "ADA-USD",
        "XRPUSDT": "XRP-USD",
        "DOTUSDT": "DOT-USD",
        "AVAXUSDT": "AVAX-USD",
        "MATICUSDT": "MATIC-USD",
        "LTCUSDT": "LTC-USD",
    }
    if symbol in mapping:
        return mapping[symbol]
    if symbol.endswith("USDT"):
        return symbol[:-4] + "-USD"
    return symbol


async def _fetch_ohlcv(symbol: str, market_type: str) -> pd.DataFrame:
    """Fetch OHLCV data: try Redis cache first, then yfinance fallback."""
    try:
        from app.redis_client import price_cache
        raw = await price_cache.get(f"ohlcv:{symbol}:1d")
        if raw:
            rows = json.loads(raw)
            if rows and len(rows) >= 20:
                df = pd.DataFrame(rows)
                if "close" in df.columns:
                    return df
    except Exception as e:
        logger.debug("Redis OHLCV fetch failed", symbol=symbol, error=str(e))

    try:
        import yfinance as yf
        yf_symbol = symbol
        if market_type == "crypto":
            yf_symbol = _map_crypto_symbol(symbol)
        df = yf.download(yf_symbol, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning("yfinance fallback failed", symbol=symbol, error=str(e))
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# BotResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class BotResult:
    fired: bool
    reason: str
    signal: str  # "buy" | "sell" | "hold" | "alert"
    orders_created: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BotEngine
# ---------------------------------------------------------------------------

class BotEngine:
    """Evaluates a bot's trigger/conditions/action."""

    async def evaluate(self, bot: Bot, db: AsyncSession) -> BotResult:
        """
        1. Fetch recent OHLCV data for bot.symbol
        2. Compute indicators needed by conditions
        3. Evaluate all conditions (AND/OR based on condition_logic)
        4. If conditions pass → execute action (paper order)
        5. Update bot stats in DB
        Returns BotResult.
        """
        try:
            return await self._evaluate_inner(bot, db)
        except Exception as exc:
            logger.error("BotEngine evaluation failed", bot_id=bot.id, error=str(exc))
            result = BotResult(fired=False, reason=f"Error: {exc}", signal="hold")
            await self._update_bot_stats(bot, db, result)
            return result

    async def _evaluate_inner(self, bot: Bot, db: AsyncSession) -> BotResult:
        df = await _fetch_ohlcv(bot.symbol, bot.market_type)
        current_price = float(df["close"].iloc[-1]) if not df.empty and "close" in df.columns else 0.0

        raw_conditions: list[dict] = bot.conditions or []
        conditions = [ConditionConfig(**c) for c in raw_conditions]

        condition_results: list[bool] = []
        for cond in conditions:
            try:
                passed = evaluate_condition(cond, df, current_price)
                condition_results.append(passed)
            except Exception as exc:
                logger.warning("Condition evaluation error", bot_id=bot.id, error=str(exc))
                condition_results.append(False)

        logic = (bot.condition_logic or "ALL").upper()
        if not condition_results:
            conditions_passed = True
        elif logic == "ANY":
            conditions_passed = any(condition_results)
        else:
            conditions_passed = all(condition_results)

        if not conditions_passed:
            result = BotResult(
                fired=False,
                reason=f"Conditions not met ({logic}: {condition_results})",
                signal="hold",
            )
            await self._update_bot_stats(bot, db, result)
            return result

        action_dict: dict = bot.action or {}
        action = ActionConfig(**action_dict)
        orders_created: list[str] = []
        signal = "hold"

        if action.type in ("open_long", "open_short"):
            signal = "buy" if action.type == "open_long" else "sell"
            order_id = await self._create_paper_order(bot, action, current_price, signal, db)
            if order_id:
                orders_created.append(order_id)
            reason = f"Action fired: {action.type} {bot.symbol} @ {current_price:.4f}"
        elif action.type == "close_position":
            signal = "sell"
            reason = f"Close position: {bot.symbol} @ {current_price:.4f}"
        elif action.type == "send_alert":
            signal = "alert"
            msg = action.alert_message or "Bot alert triggered"
            logger.info("Bot alert", bot_id=bot.id, bot_name=bot.name, message=msg, symbol=bot.symbol)
            reason = f"Alert sent: {msg}"
        elif action.type == "reduce_position":
            signal = "sell"
            reason = f"Reduce position by {action.reduce_by_pct}%"
        elif action.type == "open_option_spread":
            # Multi-leg options. Emitted as an actionable alert: the options
            # desk / TradeStation routing consumes the leg plan. Never crashes
            # if legs are malformed — degrades to a plain alert.
            legs = action.legs or []
            signal = "alert"
            if legs:
                plan = ", ".join(
                    f"{lg.side} {lg.option_type}"
                    + (f" {lg.delta:g}Δ" if lg.delta is not None else "")
                    + (f" {lg.strike:g}K" if lg.strike is not None else "")
                    + f" {lg.dte}DTE x{lg.ratio}"
                    for lg in legs
                )
                reason = f"Options spread on {bot.symbol}: {plan}"
                logger.info(
                    "Bot options spread",
                    bot_id=bot.id,
                    bot_name=bot.name,
                    symbol=bot.symbol,
                    legs=[lg.model_dump() for lg in legs],
                )
            else:
                reason = f"Options spread on {bot.symbol}: no legs configured"
                logger.warning(
                    "Bot options spread missing legs", bot_id=bot.id, bot_name=bot.name
                )
        else:
            signal = "hold"
            reason = f"Unknown action: {action.type}"

        result = BotResult(
            fired=True,
            reason=reason,
            signal=signal,
            orders_created=orders_created,
            details={
                "price": current_price,
                "conditions": condition_results,
                "logic": logic,
            },
        )
        await self._update_bot_stats(bot, db, result)
        return result

    async def _create_paper_order(
        self,
        bot: Bot,
        action: ActionConfig,
        current_price: float,
        side: str,
        db: AsyncSession,
    ) -> str | None:
        """Create a paper Order record in the DB."""
        try:
            from app.models.order import Order

            account_id = bot.account_id

            order = Order(
                id=str(uuid.uuid4()),
                account_id=account_id or "paper",
                broker_order_id="paper",
                symbol=bot.symbol,
                side=side,
                order_type="market",
                quantity=None,
                status="paper",
                raw_payload={
                    "bot_id": bot.id,
                    "bot_name": bot.name,
                    "size_pct": action.size_pct,
                    "stop_loss_pct": action.stop_loss_pct,
                    "take_profit_pct": action.take_profit_pct,
                    "trailing_stop_pct": action.trailing_stop_pct,
                    "entry_price": current_price,
                },
                take_profit_price=(
                    current_price * (1 + action.take_profit_pct / 100)
                    if action.take_profit_pct and side == "buy"
                    else current_price * (1 - action.take_profit_pct / 100)
                    if action.take_profit_pct and side == "sell"
                    else None
                ),
                stop_loss_price=(
                    current_price * (1 - action.stop_loss_pct / 100)
                    if action.stop_loss_pct and side == "buy"
                    else current_price * (1 + action.stop_loss_pct / 100)
                    if action.stop_loss_pct and side == "sell"
                    else None
                ),
                trailing_stop_pct=action.trailing_stop_pct,
                notional=None,
            )

            if account_id:
                db.add(order)
                await db.flush()
                logger.info(
                    "Paper order created",
                    bot_id=bot.id,
                    order_id=order.id,
                    symbol=bot.symbol,
                    side=side,
                    price=current_price,
                )
            else:
                logger.info(
                    "Paper order (no account, not persisted)",
                    bot_id=bot.id,
                    symbol=bot.symbol,
                    side=side,
                    price=current_price,
                )
            return order.id
        except Exception as exc:
            logger.error("Failed to create paper order", bot_id=bot.id, error=str(exc))
            return None

    async def _update_bot_stats(self, bot: Bot, db: AsyncSession, result: BotResult) -> None:
        """Persist run stats back to the Bot row."""
        try:
            bot.run_count = (bot.run_count or 0) + 1
            bot.last_run_at = datetime.now(timezone.utc)
            bot.last_signal = result.signal
            bot.last_result = {
                "fired": result.fired,
                "reason": result.reason,
                "orders": result.orders_created,
                "details": result.details,
            }
            await db.commit()
        except Exception as exc:
            logger.error("Failed to update bot stats", bot_id=bot.id, error=str(exc))
            try:
                await db.rollback()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Option Alpha-style exit checker — runs every 5 minutes via scheduler
# ---------------------------------------------------------------------------

async def _fetch_current_price(symbol: str, market_type: str = "equity") -> float | None:
    """Fetch latest close price via Redis cache or yfinance fallback."""
    try:
        from app.redis_client import price_cache
        raw = await price_cache.get(f"prices:{symbol}")
        if raw:
            data = json.loads(raw) if isinstance(raw, str) else raw
            price = data.get("last") or data.get("close") or data.get("ask")
            if price:
                return float(price)
    except Exception:
        pass

    try:
        import yfinance as yf
        yf_sym = _map_crypto_symbol(symbol) if market_type == "crypto" else symbol
        ticker = yf.Ticker(yf_sym)
        hist = ticker.history(period="2d", interval="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.debug("Price fetch failed", symbol=symbol, error=str(exc))

    return None


async def check_bot_exits(db: AsyncSession) -> int:
    """
    Scan all open bot paper orders (status='paper') with TP or SL set.
    Close positions that have hit their exit target and record a Trade.

    This is the Option Alpha-style trade history mechanism — every bot
    position gets a closed Trade record with entry/exit prices, P&L,
    hold time, and bot name stored in strategy_name.

    Returns number of exits triggered.
    """
    from app.models.order import Order
    from app.models.trade import Trade

    result = await db.execute(
        select(Order).where(
            Order.status == "paper",
            or_(
                Order.take_profit_price.isnot(None),
                Order.stop_loss_price.isnot(None),
            ),
        )
    )
    open_orders = result.scalars().all()

    if not open_orders:
        return 0

    now = datetime.now(timezone.utc)

    # Batch price fetches by symbol
    symbols: dict[str, str] = {}  # symbol → market_type
    for order in open_orders:
        raw = order.raw_payload or {}
        market_type = raw.get("market_type", "equity")
        symbols[order.symbol] = market_type

    prices: dict[str, float] = {}
    for symbol, market_type in symbols.items():
        price = await _fetch_current_price(symbol, market_type)
        if price is not None:
            prices[symbol] = price

    exits = 0
    for order in open_orders:
        current_price = prices.get(order.symbol)
        if current_price is None:
            continue

        raw = order.raw_payload or {}
        entry_price = float(raw.get("entry_price", 0))
        if entry_price <= 0:
            continue

        tp = float(order.take_profit_price) if order.take_profit_price is not None else None
        sl = float(order.stop_loss_price) if order.stop_loss_price is not None else None
        side = order.side  # "buy" | "sell"

        exit_reason: str | None = None
        exit_price = current_price

        if side == "buy":
            if tp is not None and current_price >= tp:
                exit_reason, exit_price = "take_profit", tp
            elif sl is not None and current_price <= sl:
                exit_reason, exit_price = "stop_loss", sl
        else:  # sell / short
            if tp is not None and current_price <= tp:
                exit_reason, exit_price = "take_profit", tp
            elif sl is not None and current_price >= sl:
                exit_reason, exit_price = "stop_loss", sl

        if exit_reason is None:
            # Also close if the position has been open > 7 days (safety expiry)
            opened_at = getattr(order, "created_at", None)
            if opened_at:
                age_days = (now - opened_at).total_seconds() / 86400
                if age_days > 7:
                    exit_reason = "expired"

        if exit_reason is None:
            continue

        # Compute notional and quantity
        notional = float(order.notional) if order.notional else 1000.0
        qty = notional / entry_price

        if side == "buy":
            realized_pnl = (exit_price - entry_price) * qty
        else:
            realized_pnl = (entry_price - exit_price) * qty

        opened_at = getattr(order, "created_at", now)
        hold_seconds = int((now - opened_at).total_seconds()) if opened_at else None

        trade = Trade(
            id=str(uuid.uuid4()),
            account_id=order.account_id,
            strategy_id=order.strategy_id,
            strategy_name=raw.get("bot_name"),
            symbol=order.symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=qty,
            realized_pnl=realized_pnl,
            fees=0.0,
            opened_at=opened_at,
            closed_at=now,
            hold_seconds=hold_seconds,
            raw_payload={
                "exit_reason": exit_reason,
                "bot_id": raw.get("bot_id"),
                "bot_name": raw.get("bot_name"),
                "order_id": order.id,
                "entry_price": entry_price,
                "exit_price": exit_price,
            },
        )
        db.add(trade)

        order.status = "filled"
        order.filled_qty = qty
        order.avg_fill_price = exit_price
        order.filled_at = now

        exits += 1
        logger.info(
            "Bot position closed",
            symbol=order.symbol,
            side=side,
            exit_reason=exit_reason,
            entry=entry_price,
            exit=exit_price,
            pnl=realized_pnl,
        )

    if exits > 0:
        try:
            await db.commit()
        except Exception as exc:
            logger.error("Failed to commit bot exits", error=str(exc))
            await db.rollback()
            exits = 0

    return exits
