"""
CCI reversion strategy.

The Commodity Channel Index measures how far price has stretched from its
typical-price mean in units of mean deviation. Readings below -100 mark a
statistically unusual stretch to the downside. This strategy buys the snap-back
— CCI *crossing back up* through the oversold line — but only while price is
above its trend EMA, so we fade dips inside uptrends rather than catching falling
knives. Exit when CCI reaches the overbought band (reversion complete).
"""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class CCIReversionStrategy(AbstractStrategy):
    name = "cci_reversion"
    display_name = "CCI Reversion"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    DEFAULT_PARAMS = {
        "cci_period": 20,
        "oversold": -100.0,
        "overbought": 100.0,
        "trend_period": 50,   # EMA trend filter — only buy dips in uptrends
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.cci_period = effective["cci_period"]
        self.oversold = effective["oversold"]
        self.overbought = effective["overbought"]
        self.trend_period = effective["trend_period"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if not {"high", "low", "close"}.issubset(data.columns):
            return None
        if len(data) < max(self.cci_period, self.trend_period) + 5:
            return None

        high = data["high"]
        low = data["low"]
        close = data["close"]

        cci = ta.cci(high, low, close, length=self.cci_period)
        trend_ema = ta.ema(close, length=self.trend_period)
        if cci is None or trend_ema is None:
            return None

        price = close.iloc[-1]
        trend = trend_ema.iloc[-1]
        cci_now = cci.iloc[-1]
        cci_prev = cci.iloc[-2]
        if pd.isna(trend) or pd.isna(cci_now) or pd.isna(cci_prev):
            return None

        # cross back up through the oversold line, inside an uptrend
        if price > trend and cci_prev < self.oversold <= cci_now:
            depth = min(1.0, (self.oversold - cci_prev) / 200.0)
            confidence = min(0.84, 0.58 + max(depth, 0.0) * 0.25)
            return Signal(symbol=symbol, side="buy", confidence=confidence,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket,
                          metadata={"cci": round(float(cci_now), 2), "trend": "up"})
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        cci = ta.cci(high, low, close, length=self.cci_period)
        trend_ema = ta.ema(close, length=self.trend_period)
        if cci is None or trend_ema is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # shift(1)/shift(2): entry at bar i uses only the cross seen by bar i-1
        price = close.shift(1)
        trend = trend_ema.shift(1)
        cci_s = cci.shift(1)
        cci_prev = cci.shift(2)

        entries = (price > trend) & (cci_prev < self.oversold) & (cci_s >= self.oversold)
        exits = cci_s >= self.overbought

        return BacktestSignals(entries=entries.fillna(False), exits=exits.fillna(False))
