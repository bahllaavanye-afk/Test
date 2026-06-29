"""
Donchian Channel breakout (Turtle-style trend following).

Enter long when price closes above the highest high of the last N bars (the
upper Donchian channel); exit when it closes below the lowest low of the last M
bars. Unlike the volume/ATR breakout strategy, this is a pure price-channel
trend follower with an *asymmetric* entry/exit lookback — the hallmark of the
Turtle system (slower entry, faster exit to give back less profit).
"""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class DonchianBreakoutStrategy(AbstractStrategy):
    name = "donchian_breakout"
    display_name = "Donchian Channel Breakout (Turtle)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    DEFAULT_PARAMS = {
        "entry_period": 20,  # breakout above N-bar high opens a long
        "exit_period": 10,   # close below M-bar low closes it (faster than entry)
        "atr_period": 14,    # used to size the breakout confidence
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.entry_period = effective["entry_period"]
        self.exit_period = effective["exit_period"]
        self.atr_period = effective["atr_period"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if not {"high", "low", "close"}.issubset(data.columns):
            return None
        if len(data) < self.entry_period + self.atr_period + 5:
            return None

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Upper channel = highest high of the *prior* N bars (shift(1) excludes today)
        upper = high.rolling(self.entry_period).max().shift(1)
        atr = ta.atr(high, low, close, length=self.atr_period)

        price = close.iloc[-1]
        res = upper.iloc[-1]
        if pd.isna(res):
            return None

        if price > res:
            atr_val = float(atr.iloc[-1]) if atr is not None and not pd.isna(atr.iloc[-1]) else 0.0
            # how many ATRs past the channel — bigger thrust = more conviction
            thrust = (price - res) / atr_val if atr_val > 0 else (price - res) / max(res, 1e-8)
            confidence = min(0.80, 0.56 + thrust * 0.15)
            return Signal(symbol=symbol, side="buy", confidence=confidence,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket,
                          metadata={"channel_high": round(float(res), 4),
                                    "atr": round(atr_val, 4)})
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # rolling max/min include the current bar, so shift(2) keeps the channel
        # strictly in the past relative to the close.shift(1) we compare it to.
        upper = high.rolling(self.entry_period).max().shift(2)
        lower = low.rolling(self.exit_period).min().shift(2)
        prev_close = close.shift(1)

        entries = prev_close > upper          # breakout above prior N-bar high
        exits = prev_close < lower            # breakdown below prior M-bar low

        return BacktestSignals(entries=entries.fillna(False), exits=exits.fillna(False))
