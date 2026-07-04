"""FX RSI mean-reversion (Forex desk) — fade stretched moves in ranging pairs."""

import pandas as pd

import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class FXReversionStrategy(AbstractStrategy):
    """Mean‑reversion strategy based on RSI for Forex pairs.

    Generates buy signals when the RSI falls below the oversold threshold
    and sell signals when it rises above the overbought threshold.
    """

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
        """Initialize the strategy with optional parameter overrides."""
        super().__init__(params)
        effective_params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.rsi_period = effective_params["rsi_period"]
        self.oversold = effective_params["oversold"]
        self.overbought = effective_params["overbought"]
        self.exit = effective_params["exit"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Analyze the latest data and emit a signal if RSI criteria are met.

        Returns ``None`` when there is insufficient data or no actionable signal.
        """
        if "close" not in data.columns or len(data) < self.rsi_period + 5:
            return None

        rsi = ta.rsi(data["close"], length=self.rsi_period)
        if rsi is None:
            return None

        current_rsi = rsi.iloc[-1]
        if pd.isna(current_rsi):
            return None

        if current_rsi < self.oversold:
            confidence = min(0.85, 0.55 + (self.oversold - current_rsi) / 100.0)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"rsi": round(float(current_rsi), 2)},
            )

        if current_rsi > self.overbought:
            confidence = min(0.85, 0.55 + (current_rsi - self.overbought) / 100.0)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"rsi": round(float(current_rsi), 2)},
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Generate entry and exit signals for backtesting based on RSI thresholds."""
        rsi = ta.rsi(df["close"], length=self.rsi_period)
        if rsi is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        rsi_shifted = rsi.shift(1)  # use yesterday's RSI for today's decision
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