"""Volume-confirmed price breakout above rolling high with additional trend filters."""
import pandas as pd
import numpy as np
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class BreakoutStrategy(AbstractStrategy):
    name = "breakout"
    display_name = "Volume Breakout"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    DEFAULT_PARAMS = {
        "lookback": 52,      # rolling window for resistance (bars)
        "vol_mult": 1.5,     # volume must exceed avg * vol_mult
        "atr_mult": 0.5,     # breakout distance in ATR units
        "sma_period": 20,    # short‑term trend filter
        "min_volume": 0.0,   # optional absolute volume floor
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.lookback = effective["lookback"]
        self.vol_mult = effective["vol_mult"]
        self.atr_mult = effective["atr_mult"]
        self.sma_period = effective["sma_period"]
        self.min_volume = effective["min_volume"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate a buy signal if price breaks above resistance with volume confirmation
        and additional trend filters.

        Returns:
            Signal | None: A populated Signal on a valid breakout, otherwise None.
        """
        required_cols = {"close", "high", "low"}
        if not required_cols.issubset(data.columns):
            return None

        if len(data) < max(self.lookback, self.sma_period) + 20:
            return None

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data.get("volume", pd.Series(np.nan, index=data.index))

        # Rolling resistance (previous high) and ATR
        resistance = high.rolling(self.lookback, min_periods=self.lookback).max().shift(1)
        atr = ta.atr(high, low, close, length=14)

        # Trend filter – short‑term SMA
        sma = close.rolling(self.sma_period, min_periods=self.sma_period).mean()

        # Volume average for confirmation
        vol_avg = volume.rolling(20, min_periods=1).mean()

        # Latest values (ensure they are not NaN)
        price = close.iloc[-1]
        res = resistance.iloc[-1]
        atr_val = atr.iloc[-1] if atr is not None else np.nan
        sma_val = sma.iloc[-1] if sma is not None else np.nan
        vol_curr = volume.iloc[-1] if not volume.isna().all() else np.nan
        vol_mean = vol_avg.iloc[-1] if not vol_avg.isna().all() else np.nan

        # Guard against missing data
        if np.isnan([price, res, atr_val, sma_val, vol_curr, vol_mean]).any():
            return None

        # Entry conditions
        price_break = price > res + self.atr_mult * atr_val
        trend_confirm = price > sma_val
        vol_confirm = vol_curr > max(self.vol_mult * vol_mean, self.min_volume)

        if price_break and trend_confirm and vol_confirm:
            pct_break = (price - res) / max(res, 1e-8)
            confidence = min(0.85, 0.55 + pct_break * 3)  # tighter cap
            metadata = {
                "resistance": round(res, 4),
                "atr": round(atr_val, 4),
                "sma": round(sma_val, 4),
                "volume": round(vol_curr, 2),
                "volume_avg": round(vol_mean, 2),
            }
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata=metadata,
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Generate entry and exit series for backtesting."""
        required_cols = {"close", "high", "low"}
        if not required_cols.issubset(df.columns):
            raise ValueError("Dataframe missing required price columns.")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", pd.Series(np.nan, index=df.index))

        resistance = high.rolling(self.lookback, min_periods=self.lookback).max().shift(2)
        atr = ta.atr(high, low, close, length=14)
        sma = close.rolling(self.sma_period, min_periods=self.sma_period).mean()
        vol_avg = volume.rolling(20, min_periods=1).mean()

        # Shifted to align signal on the bar after conditions are met
        price_break = close.shift(1) > resistance + self.atr_mult * atr.shift(1)
        trend_confirm = close.shift(1) > sma.shift(1)
        vol_confirm = volume.shift(1) > self.vol_mult * vol_avg.shift(1)

        entries = price_break & trend_confirm & vol_confirm

        # Exit when price falls below resistance or breaches below entry level minus ATR
        exit_price = close.shift(1) < resistance
        trailing_exit = close.shift(1) < (close.shift(2) - self.atr_mult * atr.shift(2))
        exits = exit_price | trailing_exit

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
        )