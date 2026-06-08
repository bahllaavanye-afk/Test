"""
Gap-and-Go Event-Driven Strategy.

When a stock gaps up > 2% at the open with volume > 3× average,
this is typically a catalyst-driven event (earnings, news, upgrade).
Buy the momentum continuation for the first 30 minutes.

Classic retail/institutional pattern: gap stocks with high relative volume
have strong continuation 60-70% of the time in the first 30 minutes.

Exit conditions:
  - 30-minute time stop
  - Gap fill (price comes back to previous close)
  - Trailing stop 1% below entry
"""
import numpy as np
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class EventDrivenGapStrategy(AbstractStrategy):
    name = "event_driven_gap"
    display_name = "Gap-and-Go (Event Driven)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 60.0   # minute-level for intraday

    GAP_THRESHOLD = 0.02      # >2% gap up
    VOL_MULTIPLIER = 3.0      # >3× volume vs 20-day avg
    VOL_WINDOW = 20

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.gap_threshold = p.get("gap_threshold", self.GAP_THRESHOLD)
        self.vol_multiplier = p.get("vol_multiplier", self.VOL_MULTIPLIER)
        self.vol_window = p.get("vol_window", self.VOL_WINDOW)

    def _compute_gap(self, df: pd.DataFrame) -> float:
        """Gap = (today's open - yesterday's close) / yesterday's close."""
        if "open" not in df.columns or len(df) < 2:
            return 0.0
        prev_close = float(df["close"].iloc[-2])
        today_open = float(df["open"].iloc[-1])
        return (today_open - prev_close) / prev_close if prev_close > 0 else 0.0

    def _relative_volume(self, df: pd.DataFrame) -> float:
        """Today's volume / 20-day average volume."""
        if "volume" not in df.columns or len(df) < self.vol_window + 1:
            return 1.0
        avg_vol = float(df["volume"].iloc[-(self.vol_window + 1):-1].mean())
        today_vol = float(df["volume"].iloc[-1])
        return today_vol / avg_vol if avg_vol > 0 else 1.0

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < self.vol_window + 2:
            return None
        if "close" not in data.columns:
            return None

        gap = self._compute_gap(data)
        rvol = self._relative_volume(data)

        if gap > self.gap_threshold and rvol > self.vol_multiplier:
            # Strong gap-up with volume → buy continuation
            confidence = min(0.85, 0.60 + gap * 5 + (rvol - self.vol_multiplier) * 0.02)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "gap_pct": round(gap * 100, 2),
                    "relative_volume": round(rvol, 2),
                },
            )
        elif gap < -self.gap_threshold and rvol > self.vol_multiplier:
            # Gap-down with volume → sell short continuation
            confidence = min(0.80, 0.60 + abs(gap) * 5)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "gap_pct": round(gap * 100, 2),
                    "relative_volume": round(rvol, 2),
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]

        if "open" in df.columns:
            opens = df["open"]
            gap = (opens - close.shift(1)) / close.shift(1)
        else:
            gap = close.pct_change()

        if "volume" in df.columns:
            vol = df["volume"]
            vol_avg = vol.rolling(self.vol_window).mean()
            rvol = vol / vol_avg
        else:
            rvol = pd.Series(self.vol_multiplier + 1, index=close.index)

        # Shift everything by 1 to prevent lookahead
        gap_s = gap.shift(1)
        rvol_s = rvol.shift(1)

        entries = (gap_s > self.gap_threshold) & (rvol_s > self.vol_multiplier)
        exits = gap_s < 0.0
        short_entries = (gap_s < -self.gap_threshold) & (rvol_s > self.vol_multiplier)
        short_exits = gap_s > 0.0

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )
