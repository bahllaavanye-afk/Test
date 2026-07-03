"""FX RSI mean-reversion (Forex desk) — fade stretched moves in ranging pairs."""
import logging
import time

import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals

logger = logging.getLogger(__name__)


class FXReversionStrategy(AbstractStrategy):
    name = "fx_reversion"
    display_name = "FX RSI Reversion"
    market_type = "forex"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    DEFAULT_PARAMS = {"rsi_period": 14, "oversold": 30, "overbought": 70, "exit": 50}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        self.rsi_period = eff["rsi_period"]
        self.oversold = eff["oversold"]
        self.overbought = eff["overbought"]
        self.exit = eff["exit"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        start_time = time.time()
        signal: Signal | None = None

        if "close" not in data.columns or len(data) < self.rsi_period + 5:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "FXReversionStrategy analyze skipped",
                extra={
                    "symbol": symbol,
                    "signal_generated": False,
                    "execution_ms": elapsed_ms,
                },
            )
            return None

        rsi = ta.rsi(data["close"], length=self.rsi_period)
        if rsi is None:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "FXReversionStrategy analyze no RSI",
                extra={"symbol": symbol, "signal_generated": False, "execution_ms": elapsed_ms},
            )
            return None

        v = rsi.iloc[-1]
        if pd.isna(v):
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "FXReversionStrategy analyze NaN RSI",
                extra={"symbol": symbol, "signal_generated": False, "execution_ms": elapsed_ms},
            )
            return None

        if v < self.oversold:
            conf = min(0.85, 0.55 + (self.oversold - v) / 100.0)
            signal = Signal(
                symbol=symbol,
                side="buy",
                confidence=conf,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"rsi": round(float(v), 2)},
            )
        elif v > self.overbought:
            conf = min(0.85, 0.55 + (v - self.overbought) / 100.0)
            signal = Signal(
                symbol=symbol,
                side="sell",
                confidence=conf,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"rsi": round(float(v), 2)},
            )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "FXReversionStrategy analyze completed",
            extra={
                "symbol": symbol,
                "signal_generated": signal is not None,
                "execution_ms": elapsed_ms,
                "confidence": signal.confidence if signal else None,
                "rsi": signal.metadata.get("rsi") if signal else None,
            },
        )
        return signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        start_time = time.time()
        rsi = ta.rsi(df["close"], length=self.rsi_period)
        if rsi is None:
            empty = pd.Series(False, index=df.index)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "FXReversionStrategy backtest_signals no RSI",
                extra={"execution_ms": elapsed_ms, "signal_count": 0},
            )
            return BacktestSignals(entries=empty, exits=empty)

        r = rsi.shift(1)  # decide today from yesterday's RSI
        entries = r < self.oversold
        exits = r >= self.exit
        short_entries = r > self.overbought
        short_exits = r <= self.exit

        # Count signals for monitoring
        signal_count = int(entries.sum()) + int(short_entries.sum())

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "FXReversionStrategy backtest_signals completed",
            extra={"execution_ms": elapsed_ms, "signal_count": signal_count},
        )

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )