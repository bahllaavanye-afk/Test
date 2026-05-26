"""
Bollinger Band Mean Reversion Strategy.
Enter when price touches lower/upper band; exit at middle band.
"""
import pandas as pd
import pandas_ta as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class MeanReversionStrategy(AbstractStrategy):
    name = "mean_reversion"
    display_name = "Bollinger Band Mean Reversion"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bb_period = params.get("bb_period", 20) if params else 20
        self.bb_std = params.get("bb_std", 2.0) if params else 2.0
        self.rsi_period = params.get("rsi_period", 14) if params else 14
        self.rsi_oversold = params.get("rsi_oversold", 30) if params else 30
        self.rsi_overbought = params.get("rsi_overbought", 70) if params else 70

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns or len(data) < self.bb_period + 5:
            return None

        close = data["close"]
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        rsi = ta.rsi(close, length=self.rsi_period)

        if bb is None or rsi is None:
            return None

        lower = bb[f"BBL_{self.bb_period}_{self.bb_std}"].iloc[-1]
        upper = bb[f"BBU_{self.bb_period}_{self.bb_std}"].iloc[-1]
        mid = bb[f"BBM_{self.bb_period}_{self.bb_std}"].iloc[-1]
        price = close.iloc[-1]
        rsi_val = rsi.iloc[-1]

        if price <= lower and rsi_val < self.rsi_oversold:
            pct_below = (lower - price) / lower
            confidence = min(0.88, 0.55 + pct_below * 5)
            return Signal(symbol=symbol, side="buy", confidence=confidence,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket, target_price=mid,
                          metadata={"rsi": round(rsi_val, 2), "bb_position": "lower"})

        if price >= upper and rsi_val > self.rsi_overbought:
            pct_above = (price - upper) / upper
            confidence = min(0.88, 0.55 + pct_above * 5)
            return Signal(symbol=symbol, side="sell", confidence=confidence,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket, target_price=mid,
                          metadata={"rsi": round(rsi_val, 2), "bb_position": "upper"})
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        rsi = ta.rsi(close, length=self.rsi_period)

        if bb is None or rsi is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        lower = bb[f"BBL_{self.bb_period}_{self.bb_std}"].shift(1)
        upper = bb[f"BBU_{self.bb_period}_{self.bb_std}"].shift(1)
        mid = bb[f"BBM_{self.bb_period}_{self.bb_std}"].shift(1)
        rsi_s = rsi.shift(1)
        close_s = close.shift(1)

        entries = (close_s <= lower) & (rsi_s < self.rsi_oversold)
        exits = close_s >= mid
        short_entries = (close_s >= upper) & (rsi_s > self.rsi_overbought)
        short_exits = close_s <= mid

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )
