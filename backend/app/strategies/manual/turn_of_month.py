"""Turn-of-Month Effect
=====================
Academic basis: Lakonishok & Smidt (1988) "Are Seasonal Anomalies Real?";
McConnell & Xu (2008) "Equity Returns at the Turn of the Month" — equity
returns concentrate in the window from the last trading day of the month
through the first three trading days of the next. The effect has persisted
out-of-sample for decades across indices (pension-flow / payroll rebalancing
mechanism).

Strategy: long only inside the turn-of-month window (last bar of month → 3rd
bar of new month), flat otherwise. Confidence scales with recent trend so the
ensembler can weigh it. Pure OHLCV + calendar — no network, no lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


def _in_tom_window(idx: pd.DatetimeIndex) -> pd.Series:
    """True for the last trading day of a month and the first three of the next."""
    if len(idx) == 0:
        return pd.Series(dtype=bool)
    month = pd.Series(idx.month, index=idx)
    next_month = month.shift(-1)
    last_of_month = (month != next_month) & next_month.notna()
    day_rank = month.groupby((month != month.shift(1)).cumsum()).cumcount()
    first_three = day_rank < 3
    return (last_of_month | first_three).astype(bool)


class TurnOfMonthStrategy(AbstractStrategy):
    name = "turn_of_month"
    display_name = "Turn-of-Month Effect"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    MIN_BARS = 40

    @property
    def description(self) -> str:
        return ("Long only in the turn-of-month window (last trading day → first "
                "3 of the new month) — Lakonishok-Smidt / McConnell-Xu anomaly.")

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        try:
            if data is None or len(data) < self.MIN_BARS or "close" not in data:
                return None
            window = _in_tom_window(pd.DatetimeIndex(data.index))
            if len(window) == 0 or not bool(window.iloc[-1]):
                return None
            closes = data["close"].astype(float)
            trend = float(closes.iloc[-1] / closes.iloc[-21] - 1) if len(closes) >= 21 else 0.0
            confidence = float(min(0.80, max(0.60, 0.65 + trend * 2)))
            return Signal(
                symbol=symbol, side="buy", confidence=confidence,
                strategy_name=self.name, strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"window": "turn_of_month", "trend_21d": round(trend, 4)},
            )
        except Exception:  # noqa: BLE001 — contract: never crash the caller
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        window = _in_tom_window(pd.DatetimeIndex(df.index))
        entries = (window & ~window.shift(1).fillna(False)).astype(bool)
        exits = (~window & window.shift(1).fillna(False)).astype(bool)
        return BacktestSignals(entries=entries, exits=exits)
