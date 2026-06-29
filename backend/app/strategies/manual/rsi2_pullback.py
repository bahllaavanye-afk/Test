"""
Connors-style RSI(2) pullback strategy.

Buys a short-term *oversold* dip inside a longer-term *uptrend* — the classic
Larry Connors mean-reversion edge: a very fast RSI (default length 2) flags an
exhausted pullback, but only trades when price is above its long trend SMA so
we're buying dips in things that are still going up. Exit on mean reversion
(RSI recovers or price reclaims a short SMA), not on a fixed target.
"""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class RSI2PullbackStrategy(AbstractStrategy):
    name = "rsi2_pullback"
    display_name = "RSI(2) Pullback (Connors)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    DEFAULT_PARAMS = {
        "rsi_period": 2,      # very short RSI — Connors' signature
        "rsi_buy": 10,        # deep short-term oversold
        "rsi_exit": 60,       # reversion complete
        "trend_period": 200,  # long-term trend filter (buy dips in uptrends only)
        "exit_period": 5,     # reclaiming this SMA = pullback over
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.rsi_period = effective["rsi_period"]
        self.rsi_buy = effective["rsi_buy"]
        self.rsi_exit = effective["rsi_exit"]
        self.trend_period = effective["trend_period"]
        self.exit_period = effective["exit_period"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns or len(data) < self.trend_period + 5:
            return None

        close = data["close"]
        rsi = ta.rsi(close, length=self.rsi_period)
        if rsi is None:
            return None

        trend_sma = close.rolling(self.trend_period).mean()
        exit_sma = close.rolling(self.exit_period).mean()

        price = close.iloc[-1]
        rsi_val = rsi.iloc[-1]
        trend = trend_sma.iloc[-1]
        if pd.isna(trend) or pd.isna(rsi_val):
            return None

        if price > trend and rsi_val < self.rsi_buy:
            depth = (self.rsi_buy - rsi_val) / max(self.rsi_buy, 1e-8)
            confidence = min(0.85, 0.58 + depth * 0.25)
            return Signal(symbol=symbol, side="buy", confidence=confidence,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket, target_price=exit_sma.iloc[-1],
                          metadata={"rsi2": round(float(rsi_val), 2), "trend": "up"})
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        rsi = ta.rsi(close, length=self.rsi_period)
        if rsi is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        trend_sma = close.rolling(self.trend_period).mean()
        exit_sma = close.rolling(self.exit_period).mean()

        # shift(1): a signal at bar i may only use data through bar i-1 (no lookahead)
        price = close.shift(1)
        rsi_s = rsi.shift(1)
        trend = trend_sma.shift(1)
        exit_ma = exit_sma.shift(1)

        entries = (price > trend) & (rsi_s < self.rsi_buy)
        exits = (rsi_s > self.rsi_exit) | (price > exit_ma)

        return BacktestSignals(entries=entries.fillna(False), exits=exits.fillna(False))
