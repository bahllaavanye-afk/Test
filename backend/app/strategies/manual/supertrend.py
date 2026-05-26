"""Supertrend indicator strategy — ATR-based trend following."""
import pandas as pd
import pandas_ta as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class SupertrendStrategy(AbstractStrategy):
    name = "supertrend"
    display_name = "Supertrend"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.atr_period = params.get("atr_period", 10) if params else 10
        self.multiplier = params.get("multiplier", 3.0) if params else 3.0

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < self.atr_period + 10:
            return None

        st = ta.supertrend(data["high"], data["low"], data["close"],
                           length=self.atr_period, multiplier=self.multiplier)
        if st is None:
            return None

        col_trend = f"SUPERTd_{self.atr_period}_{self.multiplier}"
        trend = st[col_trend].iloc[-1] if col_trend in st.columns else None
        prev_trend = st[col_trend].iloc[-2] if col_trend in st.columns else None

        if trend is None or prev_trend is None:
            return None

        if trend == 1 and prev_trend == -1:  # bullish flip
            return Signal(symbol=symbol, side="buy", confidence=0.72,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket, metadata={"supertrend_flip": "bullish"})
        if trend == -1 and prev_trend == 1:  # bearish flip
            return Signal(symbol=symbol, side="sell", confidence=0.72,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket, metadata={"supertrend_flip": "bearish"})
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        st = ta.supertrend(df["high"], df["low"], df["close"],
                           length=self.atr_period, multiplier=self.multiplier)
        col = f"SUPERTd_{self.atr_period}_{self.multiplier}"
        if st is None or col not in st.columns:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)
        trend = st[col].shift(1)
        prev = trend.shift(1)
        entries = (trend == 1) & (prev == -1)
        exits = (trend == -1)
        short_entries = (trend == -1) & (prev == 1)
        short_exits = trend == 1
        return BacktestSignals(
            entries=entries.fillna(False), exits=exits.fillna(False),
            short_entries=short_entries.fillna(False), short_exits=short_exits.fillna(False),
        )
