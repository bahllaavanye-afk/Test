"""FX RSI mean-reversion (Forex desk) — fade stretched moves in ranging pairs."""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class FXReversionStrategy(AbstractStrategy):
    """Mean‑reversion strategy based on RSI for forex pairs."""

    name = "fx_reversion"
    display_name = "FX RSI Reversion"
    market_type = "forex"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    DEFAULT_PARAMS = {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "exit": 50,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        self.rsi_period = eff["rsi_period"]
        self.oversold = eff["oversold"]
        self.overbought = eff["overbought"]
        self.exit = eff["exit"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate a trading signal based on the latest RSI value."""
        if "close" not in data.columns or len(data) < self.rsi_period + 5:
            return None

        rsi = ta.rsi(data["close"], length=self.rsi_period)
        if rsi is None:
            return None

        v = rsi.iloc[-1]
        if pd.isna(v):
            return None

        if v < self.oversold:
            confidence = min(0.85, 0.55 + (self.oversold - v) / 100.0)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"rsi": round(float(v), 2)},
            )

        if v > self.overbought:
            confidence = min(0.85, 0.55 + (v - self.overbought) / 100.0)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"rsi": round(float(v), 2)},
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Create back‑test entry/exit series based on historical RSI."""
        if "close" not in df.columns:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        rsi = ta.rsi(df["close"], length=self.rsi_period)
        if rsi is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        rsi_shifted = rsi.shift(1)  # decide today from yesterday's RSI
        entries = rsi_shifted < self.oversold
        exits = rsi_shifted >= self.exit
        short_entries = rsi_shifted > self.overbought
        short_exits = rsi_shifted <= self.exit

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )