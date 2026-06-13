"""
Analyst Revision Momentum (Price-Proxy Implementation)
Academic basis: Chan, Jegadeesh, Lakonishok (1996) "Momentum Strategies";
Gutierrez & Kelley (2008) "The Long-Lasting Momentum in Weekly Returns".

Direct analyst revision data requires paid APIs (FactSet, Bloomberg). This
strategy captures the same effect via observable price dynamics: analyst upgrades
and downgraded are immediately reflected in price gaps + sustained volume, and
the momentum they create persists 1–3 months post-revision.

Revision proxy signal:
  score = (21d return / 21) - (63d return / 63)   [annualised rate difference]
  normalised = score / |63d rate|

Positive normalised score > 0.25 with positive short-term return → recent positive
  revision (stock accelerating vs its own medium-term trend) → buy
Negative normalised score < -0.25 with negative short-term return → selling
  pressure → sell

Volume confirmation: above-average volume (1.5×) on the revision window increases
confidence (institutional accumulation from rating-change flows).
"""
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class AnalystRevisionMomentumStrategy(AbstractStrategy):
    name = "analyst_revision_momentum"
    display_name = "Analyst Revision Momentum (Price Proxy)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0
    confidence_threshold = 0.65

    SHORT_WINDOW  = 21   # 1 month
    LONG_WINDOW   = 63   # 3 months
    VOL_WINDOW    = 20
    THRESHOLD     = 0.25
    MIN_BARS      = 80

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < self.MIN_BARS or "close" not in data.columns:
            return None

        close  = data["close"].astype(float)
        volume = data["volume"].astype(float) if "volume" in data.columns else None

        if len(close) < self.LONG_WINDOW + 5:
            return None

        ret_short = float(close.iloc[-1] / close.iloc[-self.SHORT_WINDOW] - 1)
        ret_long  = float(close.iloc[-1] / close.iloc[-self.LONG_WINDOW]  - 1)

        long_rate = ret_long / self.LONG_WINDOW
        if abs(long_rate) < 1e-7:
            return None

        # Revision score: excess short-term vs long-term daily return rate
        revision_score = (ret_short / self.SHORT_WINDOW) - long_rate
        normalised     = revision_score / abs(long_rate)

        # Volume confirmation
        vol_confirm = 1.0
        if volume is not None and len(volume) > self.VOL_WINDOW:
            vol_avg    = float(volume.iloc[-self.VOL_WINDOW:].mean())
            vol_recent = float(volume.iloc[-5:].mean())
            vol_confirm = min(max(vol_recent / max(vol_avg, 1.0), 0.5), 3.0)

        if normalised > self.THRESHOLD and ret_short > 0:
            side = "buy"
            conf = min(0.62 + normalised * 0.14 * vol_confirm, 0.91)
        elif normalised < -self.THRESHOLD and ret_short < 0:
            side = "sell"
            conf = min(0.62 + abs(normalised) * 0.14 * vol_confirm, 0.91)
        else:
            return None

        if conf < self.confidence_threshold:
            return None

        spot = float(close.iloc[-1])
        return Signal(
            symbol=symbol,
            side=side,
            confidence=conf,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=spot,
            metadata={
                "ret_short_21d":  round(ret_short, 4),
                "ret_long_63d":   round(ret_long,  4),
                "revision_score": round(normalised, 4),
                "vol_confirm":    round(vol_confirm, 3),
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.MIN_BARS:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )
        close = df["close"].astype(float)

        ret_short  = close.pct_change(self.SHORT_WINDOW)
        ret_long   = close.pct_change(self.LONG_WINDOW)
        long_rate  = ret_long / self.LONG_WINDOW
        denom      = long_rate.abs() + 1e-7
        revision   = (ret_short / self.SHORT_WINDOW - long_rate) / denom

        entries       = ((revision.shift(1) > self.THRESHOLD)  & (ret_short.shift(1) > 0)).fillna(False)
        short_entries = ((revision.shift(1) < -self.THRESHOLD) & (ret_short.shift(1) < 0)).fillna(False)
        exits         = (revision.shift(1).abs() < 0.05).fillna(False)
        return BacktestSignals(entries=entries, exits=exits, short_entries=short_entries)
