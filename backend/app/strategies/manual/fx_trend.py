"""FX EMA-crossover trend following (Forex desk)."""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class FXTrendStrategy(AbstractStrategy):
    name = "fx_trend"
    display_name = "FX EMA Trend"
    market_type = "forex"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    DEFAULT_PARAMS = {"fast": 20, "slow": 50}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        self.fast = eff["fast"]
        self.slow = eff["slow"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns or len(data) < self.slow + 5:
            return None
        close = data["close"]
        fast = ta.ema(close, length=self.fast)
        slow = ta.ema(close, length=self.slow)
        if fast is None or slow is None:
            return None
        f, s, fp, sp = fast.iloc[-1], slow.iloc[-1], fast.iloc[-2], slow.iloc[-2]
        if any(pd.isna(x) for x in (f, s, fp, sp)):
            return None
        if f > s and fp <= sp:  # fresh golden cross
            conf = min(0.80, 0.55 + abs(f - s) / max(s, 1e-8) * 5)
            return Signal(symbol=symbol, side="buy", confidence=conf,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket,
                          metadata={"fast_ema": round(float(f), 5), "slow_ema": round(float(s), 5)})
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        fast = ta.ema(close, length=self.fast)
        slow = ta.ema(close, length=self.slow)
        if fast is None or slow is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)
        # shift(1)/shift(2): the cross must be confirmed by the prior closed bar
        f, s = fast.shift(1), slow.shift(1)
        fp, sp = fast.shift(2), slow.shift(2)
        entries = (f > s) & (fp <= sp)   # golden cross
        exits = (f < s) & (fp >= sp)     # death cross
        return BacktestSignals(entries=entries.fillna(False), exits=exits.fillna(False))
