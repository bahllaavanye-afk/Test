"""
Realized Volatility Asymmetry
Academic basis: Barndorff-Nielsen, Kinnebrock, Shephard (2010) "Measuring Downside
Risk — Realised Semivariance" and Feunou, Jahan-Parvar, Tedongap (2017) "Which
parametric model for conditional skewness?"

Upside realized semivariance = std(positive daily log-returns).
Downside realized semivariance = std(negative daily log-returns).
Ratio = upside_vol / downside_vol.

When ratio > 1.15: price making larger upward moves than downward → positive skew
  → long signal (momentum regime)
When ratio < 0.87: downside moves dominate → negative skew
  → short/avoid signal (crash risk)

Feunou et al. show the ratio predicts next-period equity index returns with
IC ≈ 0.04–0.06 at monthly horizon, Sharpe ~1.3 when combined with momentum.
"""
import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class RealizedVolAsymmetryStrategy(AbstractStrategy):
    name = "realized_vol_asymmetry"
    display_name = "Realized Volatility Asymmetry (Semivariance)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0
    confidence_threshold = 0.65

    LOOKBACK = 60          # days for semivariance window
    LONG_THRESHOLD  = 1.15 # upside/downside ratio to go long
    SHORT_THRESHOLD = 0.87 # below this → short/exit
    MIN_BARS = 30

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < self.MIN_BARS or "close" not in data.columns:
            return None

        close    = data["close"].astype(float).tail(self.LOOKBACK)
        log_rets = np.log(close).diff().dropna()
        if len(log_rets) < self.MIN_BARS:
            return None

        pos_rets = log_rets[log_rets > 0]
        neg_rets = log_rets[log_rets < 0]
        if len(pos_rets) < 5 or len(neg_rets) < 5:
            return None

        upside_vol   = float(pos_rets.std())
        downside_vol = float(neg_rets.abs().std())
        if downside_vol < 1e-9:
            return None

        ratio = upside_vol / downside_vol

        if ratio >= self.LONG_THRESHOLD:
            side = "buy"
            conf = min(0.63 + (ratio - self.LONG_THRESHOLD) * 1.5, 0.92)
        elif ratio <= self.SHORT_THRESHOLD:
            side = "sell"
            conf = min(0.63 + (self.SHORT_THRESHOLD - ratio) * 1.5, 0.92)
        else:
            return None

        if conf < self.confidence_threshold:
            return None

        spot = float(data["close"].iloc[-1])
        return Signal(
            symbol=symbol,
            side=side,
            confidence=conf,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=spot,
            metadata={
                "upside_vol":   round(upside_vol, 6),
                "downside_vol": round(downside_vol, 6),
                "ratio":        round(ratio, 4),
                "n_pos":        len(pos_rets),
                "n_neg":        len(neg_rets),
                "lookback":     self.LOOKBACK,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.LOOKBACK + 10:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )
        close    = df["close"].astype(float)
        log_rets = np.log(close).diff()

        def _ratio(window: pd.Series) -> float:
            pos = window[window > 0]
            neg = window[window < 0]
            if len(pos) < 5 or len(neg) < 5:
                return 1.0
            dv = neg.abs().std()
            return 1.0 if dv < 1e-9 else pos.std() / dv

        ratio_series = log_rets.rolling(self.LOOKBACK, min_periods=self.MIN_BARS).apply(
            _ratio, raw=False
        )
        entries       = (ratio_series.shift(1) >= self.LONG_THRESHOLD).fillna(False)
        short_entries = (ratio_series.shift(1) <= self.SHORT_THRESHOLD).fillna(False)
        exits         = (ratio_series.shift(1) < 1.05).fillna(False)
        return BacktestSignals(entries=entries, exits=exits, short_entries=short_entries)
