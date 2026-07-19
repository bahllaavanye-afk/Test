"""
Connors-style RSI(2) pullback strategy.

Buys a short-term *oversold* dip inside a longer-term *uptrend* — the classic
Larry Connors mean-reversion edge: a very fast RSI (default length 2) flags an
exhausted pullback, but only trades when price is above its long trend SMA so
we're buying dips in things that are still going up. Entry is tightened with
short‑term price and volume confirmation, and exit adds a stop‑loss on a recent
low. Exit on mean reversion (RSI recovers or price reclaims a short SMA), not on a
fixed target.
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
        "rsi_period": 2,          # very short RSI — Connors' signature
        "rsi_buy": 10,            # deep short-term oversold
        "rsi_exit": 60,           # reversion complete
        "trend_period": 200,      # long-term trend filter (buy dips in uptrends only)
        "short_sma_period": 20,   # short‑term price confirmation SMA
        "volume_period": 20,      # short‑term volume confirmation MA
        "exit_period": 5,         # reclaiming this SMA = pullback over
        "stop_loss_period": 5,    # recent low for stop‑loss
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.rsi_period = effective["rsi_period"]
        self.rsi_buy = effective["rsi_buy"]
        self.rsi_exit = effective["rsi_exit"]
        self.trend_period = effective["trend_period"]
        self.short_sma_period = effective["short_sma_period"]
        self.volume_period = effective["volume_period"]
        self.exit_period = effective["exit_period"]
        self.stop_loss_period = effective["stop_loss_period"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        required_cols = {"close", "volume"}
        if not required_cols.issubset(data.columns) or len(data) < self.trend_period + 5:
            return None

        close = data["close"]
        rsi = ta.rsi(close, length=self.rsi_period)
        if rsi is None:
            return None

        # Trend and confirmation indicators
        trend_sma = close.rolling(self.trend_period).mean()
        short_sma = close.rolling(self.short_sma_period).mean()
        volume_ma = data["volume"].rolling(self.volume_period).mean()
        exit_sma = close.rolling(self.exit_period).mean()
        stop_low = close.rolling(self.stop_loss_period).min()

        price = close.iloc[-1]
        rsi_val = rsi.iloc[-1]
        rsi_prev = rsi.shift(1).iloc[-1]
        trend = trend_sma.iloc[-1]
        short = short_sma.iloc[-1]
        vol_ma = volume_ma.iloc[-1]
        vol = data["volume"].iloc[-1]

        # Validate latest values
        if any(pd.isna(x) for x in (trend, short, vol_ma, rsi_val, rsi_prev)):
            return None

        # Tightened entry: uptrend, short‑term price above its SMA, volume above its MA,
        # RSI deep oversold and beginning to recover (rising from previous bar)
        if (
            price > trend
            and price > short
            and vol > vol_ma
            and rsi_val < self.rsi_buy
            and rsi_val > rsi_prev
        ):
            depth = (self.rsi_buy - rsi_val) / max(self.rsi_buy, 1e-8)
            confidence = min(0.90, 0.58 + depth * 0.30)  # slightly higher ceiling
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=exit_sma.iloc[-1],
                metadata={
                    "rsi2": round(float(rsi_val), 2),
                    "trend": "up",
                    "volume": round(float(vol), 2),
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        required_cols = {"close", "volume"}
        if not required_cols.issubset(df.columns):
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close = df["close"]
        rsi = ta.rsi(close, length=self.rsi_period)
        if rsi is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # Indicators with shift(1) to avoid look‑ahead bias
        price = close.shift(1)
        rsi_s = rsi.shift(1)
        rsi_prev = rsi.shift(2)
        trend_sma = close.rolling(self.trend_period).mean().shift(1)
        short_sma = close.rolling(self.short_sma_period).mean().shift(1)
        volume_ma = df["volume"].rolling(self.volume_period).mean().shift(1)
        exit_ma = close.rolling(self.exit_period).mean().shift(1)
        stop_low = close.rolling(self.stop_loss_period).min().shift(1)

        # Entry: uptrend + short‑term price + volume + RSI dip + recovery
        entries = (
            (price > trend_sma)
            & (price > short_sma)
            & (df["volume"].shift(1) > volume_ma)
            & (rsi_s < self.rsi_buy)
            & (rsi_s > rsi_prev)
        )

        # Exit: RSI rebound, price reclaims short SMA, or price falls below recent low
        exits = (rsi_s > self.rsi_exit) | (price > exit_ma) | (price < stop_low)

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
        )