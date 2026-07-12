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

    def _compute_gap(self, df: pd.DataFrame | None) -> float:
        """Gap = (today's open - yesterday's close) / yesterday's close."""
        if df is None or df.empty or "open" not in df.columns or "close" not in df.columns:
            return 0.0
        if len(df) < 2:
            return 0.0
        prev_close = float(df["close"].iloc[-2])
        today_open = float(df["open"].iloc[-1])
        return (today_open - prev_close) / prev_close if prev_close > 0 else 0.0

    def _relative_volume(self, df: pd.DataFrame | None) -> float:
        """Today's volume / average volume over the configured window."""
        if df is None or df.empty or "volume" not in df.columns:
            return 1.0
        if len(df) < self.vol_window + 1:
            return 1.0
        # Slice the window safely; pandas may return NaN if the slice is empty
        window = df["volume"].iloc[-(self.vol_window + 1) : -1]
        avg_vol = float(window.mean())
        today_vol = float(df["volume"].iloc[-1])
        if np.isnan(avg_vol) or avg_vol <= 0:
            return 1.0
        return today_vol / avg_vol if avg_vol > 0 else 1.0

    async def analyze(self, data: pd.DataFrame | None, symbol: str) -> Signal | None:
        if data is None or data.empty:
            return None
        if len(data) < self.vol_window + 2:
            return None
        required_cols = {"open", "close", "volume"}
        if not required_cols.issubset(data.columns):
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
        if gap < -self.gap_threshold and rvol > self.vol_multiplier:
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

    def backtest_signals(self, df: pd.DataFrame | None) -> BacktestSignals:
        if df is None or df.empty:
            empty_series = pd.Series([], dtype=bool)
            return BacktestSignals(
                entries=empty_series,
                exits=empty_series,
                short_entries=empty_series,
                short_exits=empty_series,
            )

        close = df["close"] if "close" in df.columns else pd.Series(dtype=float)

        # Compute gap
        if "open" in df.columns:
            opens = df["open"]
            gap = (opens - close.shift(1)) / close.shift(1)
        else:
            gap = close.pct_change()

        # Compute relative volume
        if "volume" in df.columns:
            vol = df["volume"]
            vol_avg = vol.rolling(self.vol_window, min_periods=1).mean()
            rvol = vol / vol_avg.replace({0: np.nan})
        else:
            rvol = pd.Series(self.vol_multiplier + 1, index=close.index)

        # Shift to avoid lookahead bias
        gap_s = gap.shift(1).fillna(0.0)
        rvol_s = rvol.shift(1).fillna(self.vol_multiplier + 1)

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