"""Volume-confirmed price breakout above rolling high."""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class BreakoutStrategy(AbstractStrategy):
    name = "breakout"
    display_name = "Volume Breakout"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.lookback = params.get("lookback", 52) if params else 52     # weeks
        self.vol_mult = params.get("vol_mult", 1.5) if params else 1.5   # volume must be 1.5x avg
        self.atr_mult = params.get("atr_mult", 0.5) if params else 0.5   # price must clear by 0.5 ATR

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < self.lookback + 20:
            return None

        close = data["close"]
        high = data["high"]
        volume = data.get("volume", pd.Series(dtype=float))

        resistance = high.rolling(self.lookback).max().shift(1)
        atr = ta.atr(data["high"], data["low"], close, length=14)
        vol_avg = volume.rolling(20).mean() if len(volume) > 0 else pd.Series(1, index=data.index)

        price = close.iloc[-1]
        res = resistance.iloc[-1]
        atr_val = atr.iloc[-1] if atr is not None else 0
        vol_curr = volume.iloc[-1] if len(volume) > 0 else 1
        vol_mean = vol_avg.iloc[-1] if len(volume) > 0 else 1

        if price > res + self.atr_mult * atr_val and vol_curr > self.vol_mult * vol_mean:
            pct_break = (price - res) / max(res, 1e-8)
            confidence = min(0.82, 0.55 + pct_break * 3)
            return Signal(symbol=symbol, side="buy", confidence=confidence,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket,
                          metadata={"resistance": round(res, 4), "atr": round(atr_val, 4)})
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        high = df["high"]
        volume = df.get("volume", pd.Series(1, index=df.index))
        resistance = high.rolling(self.lookback).max().shift(2)  # shift 2 to avoid lookahead
        atr = ta.atr(df["high"], df["low"], close, length=14)
        vol_avg = volume.rolling(20).mean()

        breakout = close.shift(1) > resistance + self.atr_mult * (atr.shift(1) if atr is not None else 0)
        vol_confirm = volume.shift(1) > self.vol_mult * vol_avg.shift(1)
        entries = breakout & vol_confirm
        exits = close.shift(1) < resistance  # price falls back below resistance

        return BacktestSignals(entries=entries.fillna(False), exits=exits.fillna(False))
