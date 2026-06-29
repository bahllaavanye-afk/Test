"""Commodity SMA-trend breakout (Commodities desk).

Long when a fast SMA is above a slow SMA *and* price clears the recent high —
a trend-confirmation filter on top of a breakout, suited to commodities' long
directional runs. Exits when price breaks the recent low.
"""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class CommodityTrendStrategy(AbstractStrategy):
    name = "commodity_trend"
    display_name = "Commodity SMA-Trend Breakout"
    market_type = "commodity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    DEFAULT_PARAMS = {"fast": 30, "slow": 100, "breakout": 20, "exit_break": 10}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        self.fast = eff["fast"]
        self.slow = eff["slow"]
        self.breakout = eff["breakout"]
        self.exit_break = eff["exit_break"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if not {"high", "low", "close"}.issubset(data.columns) or len(data) < self.slow + 5:
            return None
        close = data["close"]
        fast = close.rolling(self.fast).mean().iloc[-1]
        slow = close.rolling(self.slow).mean().iloc[-1]
        hi = data["high"].rolling(self.breakout).max().shift(1).iloc[-1]
        price = close.iloc[-1]
        if any(pd.isna(x) for x in (fast, slow, hi)):
            return None
        if fast > slow and price > hi:
            conf = min(0.80, 0.56 + (price - hi) / max(hi, 1e-8) * 4)
            return Signal(symbol=symbol, side="buy", confidence=conf,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket,
                          metadata={"breakout_high": round(float(hi), 4)})
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        fast = close.rolling(self.fast).mean().shift(1)
        slow = close.rolling(self.slow).mean().shift(1)
        # rolling max/min include the current bar → shift(2) keeps the channel in the past
        upper = df["high"].rolling(self.breakout).max().shift(2)
        lower = df["low"].rolling(self.exit_break).min().shift(2)
        prev_close = close.shift(1)
        entries = (fast > slow) & (prev_close > upper)
        exits = prev_close < lower
        return BacktestSignals(entries=entries.fillna(False), exits=exits.fillna(False))
