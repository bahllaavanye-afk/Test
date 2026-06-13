"""
TradingView Indicators Desk — 12 SOTA indicator combination strategies.

All strategies use only free indicators (pandas_ta_compat), no external API calls.
All backtest_signals() use shift(1) to prevent lookahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_EMPTY = lambda idx: BacktestSignals(
    entries=pd.Series(False, index=idx),
    exits=pd.Series(False, index=idx),
    short_entries=pd.Series(False, index=idx),
    short_exits=pd.Series(False, index=idx),
)


# ── 1. EMA Stack (Triple EMA 8/21/55) ─────────────────────────────────────────

class EMAStackStrategy(AbstractStrategy):
    """Triple EMA stack: 8>21>55 = bull trend. Price > EMA8 = entry."""
    name = "ema_stack_tv"
    display_name = "EMA Stack (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 60:
            return _EMPTY(df.index)
        close = df["close"]
        e8 = ta.ema(close, 8)
        e21 = ta.ema(close, 21)
        e55 = ta.ema(close, 55)
        if e8 is None or e21 is None or e55 is None:
            return _EMPTY(df.index)
        bull = (e8 > e21) & (e21 > e55) & (close > e8)
        bear = (e8 < e21) & (e21 < e55) & (close < e8)
        bull = bull.shift(1).fillna(False)
        bear = bear.shift(1).fillna(False)
        entries = bull & ~bull.shift(1).fillna(False)
        exits = ~bull
        short_entries = bear & ~bear.shift(1).fillna(False)
        short_exits = ~bear
        return BacktestSignals(
            entries=entries.fillna(False), exits=exits.fillna(False),
            short_entries=short_entries.fillna(False), short_exits=short_exits.fillna(False),
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 60:
            return None
        close = data["close"]
        e8 = ta.ema(close, 8)
        e21 = ta.ema(close, 21)
        e55 = ta.ema(close, 55)
        if e8 is None or e21 is None or e55 is None:
            return None
        bull = e8.iloc[-1] > e21.iloc[-1] > e55.iloc[-1] and close.iloc[-1] > e8.iloc[-1]
        bear = e8.iloc[-1] < e21.iloc[-1] < e55.iloc[-1] and close.iloc[-1] < e8.iloc[-1]
        if bull:
            return Signal(symbol=symbol, side="buy", confidence=0.70,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        if bear:
            return Signal(symbol=symbol, side="sell", confidence=0.70,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 2. Squeeze Pro (BB inside KC = squeeze, momentum bar = entry) ──────────────

class SqueezeProStrategy(AbstractStrategy):
    """Lazybear Squeeze Momentum: BB inside Keltner = squeeze; first histogram bar after release."""
    name = "squeeze_pro_tv"
    display_name = "Squeeze Pro (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    @staticmethod
    def _squeeze(df: pd.DataFrame):
        close = df["close"]
        high = df["high"]
        low = df["low"]
        bb = ta.bbands(close, length=20, std=2.0)
        atr_val = ta.atr(high, low, close, length=14)
        if bb is None or atr_val is None:
            return None, None
        mid = (bb["BBM_20_2.0"] if "BBM_20_2.0" in bb.columns
               else close.rolling(20).mean())
        kc_upper = mid + 1.5 * atr_val
        kc_lower = mid - 1.5 * atr_val
        bb_upper = bb.get("BBU_20_2.0", mid + 2 * close.rolling(20).std())
        bb_lower = bb.get("BBL_20_2.0", mid - 2 * close.rolling(20).std())
        in_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        # Momentum = close - midpoint of (rolling high + low average)
        delta = close - (high.rolling(20).max() + low.rolling(20).min()) / 2
        mom = delta.ewm(span=14).mean()
        return in_squeeze, mom

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 40:
            return _EMPTY(df.index)
        in_sq, mom = self._squeeze(df)
        if in_sq is None:
            return _EMPTY(df.index)
        just_released = (~in_sq) & in_sq.shift(1).fillna(True)
        entries = (just_released & (mom > 0)).shift(1).fillna(False)
        short_entries = (just_released & (mom < 0)).shift(1).fillna(False)
        exits = (mom < 0).shift(1).fillna(False)
        short_exits = (mom > 0).shift(1).fillna(False)
        return BacktestSignals(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 40:
            return None
        in_sq, mom = self._squeeze(data)
        if in_sq is None:
            return None
        if not in_sq.iloc[-1] and in_sq.iloc[-2]:
            if mom.iloc[-1] > 0:
                return Signal(symbol=symbol, side="buy", confidence=0.75,
                              strategy_name=self.name, strategy_type=self.strategy_type,
                              risk_bucket=self.risk_bucket)
            return Signal(symbol=symbol, side="sell", confidence=0.75,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 3. WaveTrend Oscillator ────────────────────────────────────────────────────

class WaveTrendStrategy(AbstractStrategy):
    """WaveTrend: wt1 crosses wt2 in oversold/overbought zones."""
    name = "wave_trend_tv"
    display_name = "WaveTrend (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    @staticmethod
    def _wavetrend(df: pd.DataFrame, n1: int = 10, n2: int = 21):
        hlc3 = (df["high"] + df["low"] + df["close"]) / 3
        esa = hlc3.ewm(span=n1).mean()
        d = (hlc3 - esa).abs().ewm(span=n1).mean()
        ci = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
        wt1 = ci.ewm(span=n2).mean()
        wt2 = wt1.rolling(4).mean()
        return wt1, wt2

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 30:
            return _EMPTY(df.index)
        wt1, wt2 = self._wavetrend(df)
        cross_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
        cross_dn = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
        entries = (cross_up & (wt1 < -60)).shift(1).fillna(False)
        short_entries = (cross_dn & (wt1 > 60)).shift(1).fillna(False)
        exits = (wt1 > 60).shift(1).fillna(False)
        short_exits = (wt1 < -60).shift(1).fillna(False)
        return BacktestSignals(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 30:
            return None
        wt1, wt2 = self._wavetrend(data)
        cross_up = wt1.iloc[-1] > wt2.iloc[-1] and wt1.iloc[-2] <= wt2.iloc[-2]
        cross_dn = wt1.iloc[-1] < wt2.iloc[-1] and wt1.iloc[-2] >= wt2.iloc[-2]
        if cross_up and wt1.iloc[-1] < -60:
            return Signal(symbol=symbol, side="buy", confidence=0.72,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        if cross_dn and wt1.iloc[-1] > 60:
            return Signal(symbol=symbol, side="sell", confidence=0.72,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 4. Hull Suite (HMA trend direction) ───────────────────────────────────────

class HullSuiteStrategy(AbstractStrategy):
    """Hull MA(55): long when HMA turning up, short when turning down."""
    name = "hull_suite_tv"
    display_name = "Hull Suite (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    @staticmethod
    def _hma(close: pd.Series, length: int = 55) -> pd.Series:
        half = max(length // 2, 2)
        sqrt_len = max(int(np.sqrt(length)), 2)
        wma_half = close.rolling(half).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True)
        wma_full = close.rolling(length).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True)
        diff = 2 * wma_half - wma_full
        hma = diff.rolling(sqrt_len).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True)
        return hma

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 60:
            return _EMPTY(df.index)
        hma = self._hma(df["close"])
        turning_up = (hma > hma.shift(1)).shift(1).fillna(False)
        turning_dn = (hma < hma.shift(1)).shift(1).fillna(False)
        entries = turning_up & ~turning_up.shift(1).fillna(False)
        short_entries = turning_dn & ~turning_dn.shift(1).fillna(False)
        return BacktestSignals(
            entries=entries.fillna(False), exits=turning_dn.fillna(False),
            short_entries=short_entries.fillna(False), short_exits=turning_up.fillna(False),
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 60:
            return None
        hma = self._hma(data["close"])
        up = hma.iloc[-1] > hma.iloc[-2]
        was_up = hma.iloc[-2] > hma.iloc[-3]
        if up and not was_up:
            return Signal(symbol=symbol, side="buy", confidence=0.68,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        if not up and was_up:
            return Signal(symbol=symbol, side="sell", confidence=0.68,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 5. Supertrend + RSI Combo ──────────────────────────────────────────────────

class SupertrendRsiComboStrategy(AbstractStrategy):
    """Supertrend bull + RSI 40-70 = long. Supertrend bear + RSI 30-60 = short."""
    name = "supertrend_rsi_tv"
    display_name = "Supertrend + RSI (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 30:
            return _EMPTY(df.index)
        st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        rsi_val = ta.rsi(df["close"], length=14)
        if st is None or rsi_val is None:
            return _EMPTY(df.index)
        col = "SUPERTd_10_3.0"
        if col not in st.columns:
            return _EMPTY(df.index)
        trend = st[col].shift(1).fillna(0)
        rsi_s = rsi_val.shift(1).fillna(50)
        entries = ((trend == 1) & (rsi_s >= 40) & (rsi_s <= 70)).fillna(False)
        short_entries = ((trend == -1) & (rsi_s >= 30) & (rsi_s <= 60)).fillna(False)
        exits = (trend == -1).fillna(False)
        short_exits = (trend == 1).fillna(False)
        return BacktestSignals(entries=entries, exits=exits,
                               short_entries=short_entries, short_exits=short_exits)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 30:
            return None
        st = ta.supertrend(data["high"], data["low"], data["close"], length=10, multiplier=3.0)
        rsi_val = ta.rsi(data["close"], length=14)
        if st is None or rsi_val is None:
            return None
        col = "SUPERTd_10_3.0"
        if col not in st.columns:
            return None
        trend = st[col].iloc[-1]
        rsi_now = rsi_val.iloc[-1]
        if trend == 1 and 40 <= rsi_now <= 70:
            return Signal(symbol=symbol, side="buy", confidence=0.73,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        if trend == -1 and 30 <= rsi_now <= 60:
            return Signal(symbol=symbol, side="sell", confidence=0.73,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 6. KAMA + ROC ─────────────────────────────────────────────────────────────

class KamaRocStrategy(AbstractStrategy):
    """Kaufman AMA(10) + ROC(10): cross above + positive ROC = long."""
    name = "kama_roc_tv"
    display_name = "KAMA + ROC (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    @staticmethod
    def _kama(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        kama = close.copy().astype(float)
        for i in range(n, len(close)):
            direction = abs(close.iloc[i] - close.iloc[i - n])
            volatility = sum(abs(close.iloc[j] - close.iloc[j - 1])
                             for j in range(i - n + 1, i + 1))
            er = direction / volatility if volatility != 0 else 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama.iloc[i] = kama.iloc[i - 1] + sc * (close.iloc[i] - kama.iloc[i - 1])
        return kama

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 20:
            return _EMPTY(df.index)
        close = df["close"]
        kama = self._kama(close)
        roc = close.pct_change(10) * 100
        cross_up = (close > kama) & (close.shift(1) <= kama.shift(1))
        cross_dn = (close < kama) & (close.shift(1) >= kama.shift(1))
        entries = (cross_up & (roc > 0)).shift(1).fillna(False)
        short_entries = (cross_dn & (roc < 0)).shift(1).fillna(False)
        exits = (close < kama).shift(1).fillna(False)
        short_exits = (close > kama).shift(1).fillna(False)
        return BacktestSignals(entries=entries, exits=exits,
                               short_entries=short_entries, short_exits=short_exits)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 20:
            return None
        close = data["close"]
        kama = self._kama(close)
        roc = close.pct_change(10).iloc[-1] * 100
        cross_up = close.iloc[-1] > kama.iloc[-1] and close.iloc[-2] <= kama.iloc[-2]
        cross_dn = close.iloc[-1] < kama.iloc[-1] and close.iloc[-2] >= kama.iloc[-2]
        if cross_up and roc > 0:
            return Signal(symbol=symbol, side="buy", confidence=0.68,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        if cross_dn and roc < 0:
            return Signal(symbol=symbol, side="sell", confidence=0.68,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 7. VWAP Bands (mean reversion) ────────────────────────────────────────────

class VwapBandsStrategy(AbstractStrategy):
    """Intraday VWAP ± 2σ: revert from bands toward VWAP."""
    name = "vwap_bands_tv"
    display_name = "VWAP Bands (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 60.0

    @staticmethod
    def _vwap_bands(df: pd.DataFrame, std_mult: float = 2.0):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"].replace(0, np.nan)
        cum_vol = vol.cumsum()
        cum_tp_vol = (tp * vol).cumsum()
        vwap = cum_tp_vol / cum_vol
        variance = ((tp - vwap) ** 2 * vol).cumsum() / cum_vol
        sigma = np.sqrt(variance.clip(lower=0))
        return vwap, vwap + std_mult * sigma, vwap - std_mult * sigma

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 20:
            return _EMPTY(df.index)
        vwap, upper, lower = self._vwap_bands(df)
        close = df["close"]
        touched_lower = close <= lower
        touched_upper = close >= upper
        entries = touched_lower.shift(1).fillna(False)
        exits = (close >= vwap).shift(1).fillna(False)
        short_entries = touched_upper.shift(1).fillna(False)
        short_exits = (close <= vwap).shift(1).fillna(False)
        return BacktestSignals(entries=entries, exits=exits,
                               short_entries=short_entries, short_exits=short_exits)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 20:
            return None
        vwap, upper, lower = self._vwap_bands(data)
        close = data["close"].iloc[-1]
        if close <= lower.iloc[-1]:
            return Signal(symbol=symbol, side="buy", confidence=0.65,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket,
                          target_price=float(vwap.iloc[-1]))
        if close >= upper.iloc[-1]:
            return Signal(symbol=symbol, side="sell", confidence=0.65,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket,
                          target_price=float(vwap.iloc[-1]))
        return None


# ── 8. Ichimoku Cloud ──────────────────────────────────────────────────────────

class IchimokuCloudStrategy(AbstractStrategy):
    """Ichimoku: price > cloud AND tenkan > kijun = bull. Price < cloud = bear."""
    name = "ichimoku_cloud_tv"
    display_name = "Ichimoku Cloud (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    @staticmethod
    def _ichimoku(df: pd.DataFrame):
        h, l = df["high"], df["low"]
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        return tenkan, kijun, senkou_a, senkou_b

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 80:
            return _EMPTY(df.index)
        tenkan, kijun, sa, sb = self._ichimoku(df)
        close = df["close"]
        cloud_top = pd.concat([sa, sb], axis=1).max(axis=1)
        cloud_bot = pd.concat([sa, sb], axis=1).min(axis=1)
        bull = (close > cloud_top) & (tenkan > kijun)
        bear = (close < cloud_bot) & (tenkan < kijun)
        entries = (bull & ~bull.shift(1).fillna(False)).shift(1).fillna(False)
        short_entries = (bear & ~bear.shift(1).fillna(False)).shift(1).fillna(False)
        exits = (close < cloud_bot).shift(1).fillna(False)
        short_exits = (close > cloud_top).shift(1).fillna(False)
        return BacktestSignals(entries=entries, exits=exits,
                               short_entries=short_entries, short_exits=short_exits)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 80:
            return None
        tenkan, kijun, sa, sb = self._ichimoku(data)
        close = data["close"].iloc[-1]
        cloud_top = max(sa.iloc[-1], sb.iloc[-1]) if not (np.isnan(sa.iloc[-1]) or np.isnan(sb.iloc[-1])) else None
        cloud_bot = min(sa.iloc[-1], sb.iloc[-1]) if cloud_top is not None else None
        if cloud_top is None:
            return None
        if close > cloud_top and tenkan.iloc[-1] > kijun.iloc[-1]:
            return Signal(symbol=symbol, side="buy", confidence=0.74,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        if close < cloud_bot and tenkan.iloc[-1] < kijun.iloc[-1]:
            return Signal(symbol=symbol, side="sell", confidence=0.74,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 9. MACD Divergence ─────────────────────────────────────────────────────────

class MacdDivergenceStrategy(AbstractStrategy):
    """MACD divergence: bullish = price lower low, MACD higher low."""
    name = "macd_divergence_tv"
    display_name = "MACD Divergence (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    @staticmethod
    def _find_divergence(close: pd.Series, hist: pd.Series, window: int = 20):
        bull_div = pd.Series(False, index=close.index)
        bear_div = pd.Series(False, index=close.index)
        for i in range(window, len(close)):
            seg_close = close.iloc[i - window:i + 1]
            seg_hist = hist.iloc[i - window:i + 1]
            if seg_close.iloc[-1] < seg_close.min() * 1.01:  # near low
                if seg_hist.iloc[-1] > seg_hist.iloc[seg_hist.argmin()]:
                    bull_div.iloc[i] = True
            if seg_close.iloc[-1] > seg_close.max() * 0.99:  # near high
                if seg_hist.iloc[-1] < seg_hist.iloc[seg_hist.argmax()]:
                    bear_div.iloc[i] = True
        return bull_div, bear_div

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 50:
            return _EMPTY(df.index)
        macd_df = ta.macd(df["close"])
        if macd_df is None:
            return _EMPTY(df.index)
        hist_col = [c for c in macd_df.columns if "h" in c.lower() or "hist" in c.lower()]
        if not hist_col:
            return _EMPTY(df.index)
        hist = macd_df[hist_col[0]]
        bull, bear = self._find_divergence(df["close"], hist)
        entries = bull.shift(1).fillna(False)
        short_entries = bear.shift(1).fillna(False)
        exits = bear.shift(1).fillna(False)
        short_exits = bull.shift(1).fillna(False)
        return BacktestSignals(entries=entries, exits=exits,
                               short_entries=short_entries, short_exits=short_exits)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 50:
            return None
        macd_df = ta.macd(data["close"])
        if macd_df is None:
            return None
        hist_col = [c for c in macd_df.columns if "h" in c.lower()]
        if not hist_col:
            return None
        hist = macd_df[hist_col[0]]
        bull, bear = self._find_divergence(data["close"], hist, window=20)
        if bull.iloc[-1]:
            return Signal(symbol=symbol, side="buy", confidence=0.71,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        if bear.iloc[-1]:
            return Signal(symbol=symbol, side="sell", confidence=0.71,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 10. ADX + DMI ─────────────────────────────────────────────────────────────

class AdxDmiStrategy(AbstractStrategy):
    """ADX>25 = trending. DI+>DI- = long. DI->DI+ = short. ADX<20 = no trade."""
    name = "adx_dmi_tv"
    display_name = "ADX + DMI (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 30:
            return _EMPTY(df.index)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is None:
            return _EMPTY(df.index)
        adx_col = [c for c in adx_df.columns if c.upper().startswith("ADX")]
        dmp_col = [c for c in adx_df.columns if "DMP" in c.upper() or "DI+" in c]
        dmn_col = [c for c in adx_df.columns if "DMN" in c.upper() or "DI-" in c]
        if not adx_col or not dmp_col or not dmn_col:
            return _EMPTY(df.index)
        adx = adx_df[adx_col[0]].shift(1).fillna(0)
        dmp = adx_df[dmp_col[0]].shift(1).fillna(0)
        dmn = adx_df[dmn_col[0]].shift(1).fillna(0)
        trending = adx > 25
        entries = (trending & (dmp > dmn) & ~(dmp.shift(1) > dmn.shift(1))).fillna(False)
        short_entries = (trending & (dmn > dmp) & ~(dmn.shift(1) > dmp.shift(1))).fillna(False)
        exits = ((adx < 20) | (dmn > dmp)).fillna(False)
        short_exits = ((adx < 20) | (dmp > dmn)).fillna(False)
        return BacktestSignals(entries=entries, exits=exits,
                               short_entries=short_entries, short_exits=short_exits)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 30:
            return None
        adx_df = ta.adx(data["high"], data["low"], data["close"], length=14)
        if adx_df is None:
            return None
        adx_col = [c for c in adx_df.columns if c.upper().startswith("ADX")]
        dmp_col = [c for c in adx_df.columns if "DMP" in c.upper()]
        dmn_col = [c for c in adx_df.columns if "DMN" in c.upper()]
        if not adx_col or not dmp_col or not dmn_col:
            return None
        adx_v = adx_df[adx_col[0]].iloc[-1]
        dmp_v = adx_df[dmp_col[0]].iloc[-1]
        dmn_v = adx_df[dmn_col[0]].iloc[-1]
        if adx_v > 25 and dmp_v > dmn_v:
            return Signal(symbol=symbol, side="buy", confidence=0.70,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        if adx_v > 25 and dmn_v > dmp_v:
            return Signal(symbol=symbol, side="sell", confidence=0.70,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 11. Stoch RSI + MACD ──────────────────────────────────────────────────────

class StochRsiMacdStrategy(AbstractStrategy):
    """StochRSI K>D + MACD>Signal in oversold/overbought = entry."""
    name = "stoch_rsi_macd_tv"
    display_name = "StochRSI + MACD (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    @staticmethod
    def _stoch_rsi(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14, k: int = 3):
        rsi_val = close.diff().pipe(lambda d: d.clip(lower=0).ewm(alpha=1/rsi_len).mean() /
                                    (d.clip(lower=0).ewm(alpha=1/rsi_len).mean() +
                                     (-d).clip(lower=0).ewm(alpha=1/rsi_len).mean() + 1e-10) * 100)
        rsi_min = rsi_val.rolling(stoch_len).min()
        rsi_max = rsi_val.rolling(stoch_len).max()
        k_line = ((rsi_val - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100).rolling(k).mean()
        d_line = k_line.rolling(3).mean()
        return k_line, d_line

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 40:
            return _EMPTY(df.index)
        k, d = self._stoch_rsi(df["close"])
        macd_df = ta.macd(df["close"])
        if macd_df is None:
            return _EMPTY(df.index)
        macd_col = [c for c in macd_df.columns if c.upper().startswith("MACD_")]
        sig_col = [c for c in macd_df.columns if "MACD" in c.upper() and "S" in c.upper()]
        if not macd_col or not sig_col:
            return _EMPTY(df.index)
        macd_v = macd_df[macd_col[0]]
        sig_v = macd_df[sig_col[0]]
        k_s = k.shift(1).fillna(50)
        d_s = d.shift(1).fillna(50)
        macd_s = macd_v.shift(1).fillna(0)
        sig_s = sig_v.shift(1).fillna(0)
        bull = (k_s > d_s) & (macd_s > sig_s) & (k_s < 20)
        bear = (k_s < d_s) & (macd_s < sig_s) & (k_s > 80)
        entries = (bull & ~bull.shift(1).fillna(False)).fillna(False)
        short_entries = (bear & ~bear.shift(1).fillna(False)).fillna(False)
        exits = (k_s > 80).fillna(False)
        short_exits = (k_s < 20).fillna(False)
        return BacktestSignals(entries=entries, exits=exits,
                               short_entries=short_entries, short_exits=short_exits)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 40:
            return None
        k, d = self._stoch_rsi(data["close"])
        macd_df = ta.macd(data["close"])
        if macd_df is None:
            return None
        macd_col = [c for c in macd_df.columns if c.upper().startswith("MACD_")]
        sig_col = [c for c in macd_df.columns if "MACD" in c.upper() and "S" in c.upper()]
        if not macd_col or not sig_col:
            return None
        k_v, d_v = k.iloc[-1], d.iloc[-1]
        macd_v = macd_df[macd_col[0]].iloc[-1]
        sig_v = macd_df[sig_col[0]].iloc[-1]
        if k_v > d_v and macd_v > sig_v and k_v < 20:
            return Signal(symbol=symbol, side="buy", confidence=0.72,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        if k_v < d_v and macd_v < sig_v and k_v > 80:
            return Signal(symbol=symbol, side="sell", confidence=0.72,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket)
        return None


# ── 12. Elliott Wave Proxy (ZigZag + Fibonacci) ───────────────────────────────

class ElliottWaveProxyStrategy(AbstractStrategy):
    """Elliott Wave proxy: ZigZag pivots + 61.8% Fib retracement for wave 3 entry."""
    name = "elliott_wave_proxy_tv"
    display_name = "Elliott Wave Proxy (TV)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 14400.0

    @staticmethod
    def _zigzag_pivots(close: pd.Series, pct: float = 0.05) -> list[tuple[int, float, str]]:
        pivots: list[tuple[int, float, str]] = []
        direction = 0
        last_idx, last_price = 0, float(close.iloc[0])
        for i in range(1, len(close)):
            price = float(close.iloc[i])
            if direction == 0:
                direction = 1 if price > last_price else -1
            if direction == 1 and price < last_price * (1 - pct):
                pivots.append((last_idx, last_price, "H"))
                direction = -1
                last_idx, last_price = i, price
            elif direction == -1 and price > last_price * (1 + pct):
                pivots.append((last_idx, last_price, "L"))
                direction = 1
                last_idx, last_price = i, price
            elif (direction == 1 and price > last_price) or (direction == -1 and price < last_price):
                last_idx, last_price = i, price
        return pivots

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 60:
            return _EMPTY(df.index)
        close = df["close"]
        entries = pd.Series(False, index=df.index)
        short_entries = pd.Series(False, index=df.index)
        try:
            pivots = self._zigzag_pivots(close)
            if len(pivots) >= 4:
                for i in range(2, len(pivots) - 1):
                    p0, p1, p2, p3 = pivots[i-2], pivots[i-1], pivots[i], pivots[i+1]
                    if p0[2] == "L" and p1[2] == "H" and p2[2] == "L":
                        wave1 = p1[1] - p0[1]
                        retrace = p1[1] - p2[1]
                        fib_level = retrace / wave1 if wave1 > 0 else 0
                        if 0.50 <= fib_level <= 0.786:
                            idx_val = df.index[min(p3[0] + 1, len(df) - 1)]
                            entries.loc[idx_val] = True
        except Exception:
            pass
        entries = entries.shift(1).fillna(False)
        exits = pd.Series(False, index=df.index)
        for i in range(2, len(df)):
            if entries.iloc[i - 1]:
                exits.iloc[i] = True
        return BacktestSignals(entries=entries, exits=exits,
                               short_entries=short_entries, short_exits=short_entries)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 60:
            return None
        pivots = self._zigzag_pivots(data["close"])
        if len(pivots) < 3:
            return None
        p0, p1, p2 = pivots[-3], pivots[-2], pivots[-1]
        if p0[2] == "L" and p1[2] == "H" and p2[2] == "L":
            wave1 = p1[1] - p0[1]
            retrace = (p1[1] - p2[1]) / wave1 if wave1 > 0 else 0
            if 0.50 <= retrace <= 0.786:
                return Signal(symbol=symbol, side="buy", confidence=0.66,
                              strategy_name=self.name, strategy_type=self.strategy_type,
                              risk_bucket=self.risk_bucket,
                              metadata={"fib_level": round(retrace, 3)})
        return None
