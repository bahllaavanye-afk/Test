"""Opening Gap Fade (gap-fill mean reversion)
==========================================
Academic basis: overnight gaps against the prevailing trend tend to revert
("gap fill") — documented in the intraday-seasonality literature and by
practitioner studies on index ETFs (e.g. Plastun et al. 2020 on price-gap
anomalies). Down-gaps in non-crisis regimes revert most reliably; large gaps
WITH the trend continue (skip those).

Strategy on daily bars: yesterday's close vs today's open gap ≥ GAP_MIN —
fade it (buy a down-gap, sell an up-gap) when 20d realized vol is not in
crisis territory. Confidence scales with gap size up to a cap. Pure OHLCV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class GapFillFadeStrategy(AbstractStrategy):
    name = "gap_fill_fade"
    display_name = "Opening Gap Fade"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    GAP_MIN = 0.004     # 0.4% minimum gap — below that it's noise
    GAP_MAX = 0.03      # >3% gaps are news, not noise — don't fade
    VOL_CRISIS = 0.40   # skip when 20d annualized vol above this
    MIN_BARS = 30

    @property
    def description(self) -> str:
        return ("Fade 0.4-3% overnight gaps on daily bars outside crisis vol — "
                "gap-fill mean reversion (Plastun et al.).")

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        try:
            if data is None or len(data) < self.MIN_BARS:
                return None
            if not {"open", "close"}.issubset(data.columns):
                return None
            opens = data["open"].astype(float)
            closes = data["close"].astype(float)
            prev_close = float(closes.iloc[-2])
            today_open = float(opens.iloc[-1])
            if prev_close <= 0 or today_open <= 0:
                return None
            gap = today_open / prev_close - 1
            if not (self.GAP_MIN <= abs(gap) <= self.GAP_MAX):
                return None
            rets = np.log(closes / closes.shift(1)).dropna().tail(20)
            vol = float(rets.std() * np.sqrt(252)) if len(rets) >= 10 else 0.0
            if vol > self.VOL_CRISIS:
                return None                     # crisis gaps trend, don't fade
            side = "buy" if gap < 0 else "sell"
            confidence = float(min(0.78, 0.60 + abs(gap) * 8))
            return Signal(
                symbol=symbol, side=side, confidence=confidence,
                strategy_name=self.name, strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"gap_pct": round(gap * 100, 3), "vol_20d": round(vol, 3)},
            )
        except Exception:  # noqa: BLE001 — contract: never crash the caller
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        opens = df["open"].astype(float)
        closes = df["close"].astype(float)
        gap = opens / closes.shift(1) - 1
        fade_long = (gap <= -self.GAP_MIN) & (gap >= -self.GAP_MAX)
        fade_short = (gap >= self.GAP_MIN) & (gap <= self.GAP_MAX)
        entries = fade_long.shift(1).fillna(False).astype(bool)
        exits = entries.shift(2).fillna(False).astype(bool)      # ~2-bar hold
        short_entries = fade_short.shift(1).fillna(False).astype(bool)
        short_exits = short_entries.shift(2).fillna(False).astype(bool)
        return BacktestSignals(entries=entries, exits=exits,
                               short_entries=short_entries, short_exits=short_exits)
