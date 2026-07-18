"""Connors Double-7
=================
Academic/practitioner basis: Connors & Alvarez, "Short Term Trading Strategies
That Work" (2009) — the Double-7: in a long-term uptrend (close > 200-SMA),
buy a 7-day closing low, exit on a 7-day closing high. One of the most robust
published mean-reversion rules on index ETFs; complements rsi2_pullback
(same family, different trigger).

Pure OHLCV, long-only, trend-filtered. Confidence scales with how stretched
the pullback is relative to 20d vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class DoubleSevenStrategy(AbstractStrategy):
    name = "double_seven"
    display_name = "Connors Double-7"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    LOOKBACK = 7
    TREND_SMA = 200
    MIN_BARS = 210

    @property
    def description(self) -> str:
        return ("Connors Double-7: buy 7-day closing lows above the 200-SMA, "
                "exit on 7-day closing highs.")

    async def analyze(self, data: pd.DataFrame, symbol: str = "SPY") -> Signal | None:
        try:
            if data is None or len(data) < self.MIN_BARS or "close" not in data:
                return None
            closes = data["close"].astype(float)
            last = float(closes.iloc[-1])
            sma200 = float(closes.tail(self.TREND_SMA).mean())
            if last <= sma200:
                return None                       # only trade with the trend
            window = closes.tail(self.LOOKBACK)
            if last > float(window.min()) + 1e-12:
                return None                       # not a fresh 7-day low
            rets = np.log(closes / closes.shift(1)).dropna().tail(20)
            vol = float(rets.std()) if len(rets) >= 10 else 0.01
            stretch = (float(window.max()) - last) / max(last * vol, 1e-9)
            confidence = float(min(0.80, 0.62 + min(stretch, 3.0) * 0.05))
            return Signal(
                symbol=symbol, side="buy", confidence=confidence,
                strategy_name=self.name, strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"sma200": round(sma200, 2), "stretch_sigmas": round(stretch, 2)},
            )
        except Exception:  # noqa: BLE001 — contract: never crash the caller
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        closes = df["close"].astype(float)
        sma = closes.rolling(self.TREND_SMA).mean()
        low7 = closes.rolling(self.LOOKBACK).min()
        high7 = closes.rolling(self.LOOKBACK).max()
        entries = ((closes <= low7) & (closes > sma)).shift(1).fillna(False).astype(bool)
        exits = (closes >= high7).shift(1).fillna(False).astype(bool)
        return BacktestSignals(entries=entries, exits=exits)
