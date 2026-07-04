"""FX RSI mean-reversion (Forex desk) — fade stretched moves in ranging pairs."""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class FXReversionStrategy(AbstractStrategy):
    name = "fx_reversion"
    display_name = "FX RSI Reversion"
    market_type = "forex"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    DEFAULT_PARAMS = {"rsi_period": 14, "oversold": 30, "overbought": 70, "exit": 50}

    def __init__(self, params: dict | None = None):
        if params is not None and not isinstance(params, dict):
            raise ValueError("params must be a dict or None")
        # Ensure only known parameter keys are provided
        unknown_keys = set(params or {}) - set(self.DEFAULT_PARAMS)
        if unknown_keys:
            raise ValueError(f"Unknown parameter keys: {', '.join(sorted(unknown_keys))}")
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        # Validate numeric parameters
        if not isinstance(eff["rsi_period"], int) or eff["rsi_period"] <= 0:
            raise ValueError("rsi_period must be a positive integer")
        for key in ("oversold", "overbought", "exit"):
            if not isinstance(eff[key], (int, float)):
                raise ValueError(f"{key} must be a numeric value")
        self.rsi_period = eff["rsi_period"]
        self.oversold = eff["oversold"]
        self.overbought = eff["overbought"]
        self.exit = eff["exit"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")
        if "close" not in data.columns:
            raise ValueError("data must contain a 'close' column")
        if len(data) < self.rsi_period + 5:
            return None
        rsi = ta.rsi(data["close"], length=self.rsi_period)
        if rsi is None:
            return None
        v = rsi.iloc[-1]
        if pd.isna(v):
            return None
        if v < self.oversold:
            conf = min(0.85, 0.55 + (self.oversold - v) / 100.0)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=conf,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"rsi": round(float(v), 2)},
            )
        if v > self.overbought:
            conf = min(0.85, 0.55 + (v - self.overbought) / 100.0)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=conf,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"rsi": round(float(v), 2)},
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if "close" not in df.columns:
            raise ValueError("df must contain a 'close' column")
        rsi = ta.rsi(df["close"], length=self.rsi_period)
        if rsi is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)
        r = rsi.shift(1)  # decide today from yesterday's RSI
        entries = r < self.oversold
        exits = r >= self.exit
        short_entries = r > self.overbought
        short_exits = r <= self.exit
        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )