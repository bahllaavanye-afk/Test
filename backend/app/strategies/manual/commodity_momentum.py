"""Commodity time-series momentum (Commodities desk).

Classic managed-futures edge: be long while the asset's own trailing return is
positive, flat/short when it turns negative. Computed on lagged prices so the
signal at bar t uses only data through t‑1. This version tightens entry
conditions with additional trend and volume confirmations and refines the exit
logic.
"""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class CommodityMomentumStrategy(AbstractStrategy):
    name = "commodity_momentum"
    display_name = "Commodity Time-Series Momentum"
    market_type = "commodity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    # Default parameters – can be overridden per‑instance
    DEFAULT_PARAMS = {
        "lookback": 60,               # momentum look‑back period (bars)
        "mom_threshold": 0.002,       # minimum momentum (0.2 %) for entry
        "ma_period": 20,              # moving‑average period for trend filter
        "vol_lookback": 20,           # volume look‑back for filter
        "vol_min_factor": 0.5,        # require current volume > factor * avg volume
        "confirm_periods": 2,         # consecutive positive momentum periods required
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        self.lookback = eff["lookback"]
        self.mom_threshold = eff["mom_threshold"]
        self.ma_period = eff["ma_period"]
        self.vol_lookback = eff["vol_lookback"]
        self.vol_min_factor = eff["vol_min_factor"]
        self.confirm_periods = eff["confirm_periods"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate a buy signal if momentum and confirmation filters are satisfied."""
        required_cols = {"close"}
        if not required_cols.issubset(data.columns):
            return None
        if len(data) < max(self.lookback, self.ma_period, self.vol_lookback) + self.confirm_periods:
            return None

        close = data["close"]
        # Momentum calculation based on the most recent completed bar
        mom = close.iloc[-1] / close.iloc[-1 - self.lookback] - 1.0
        if pd.isna(mom) or mom <= self.mom_threshold:
            return None

        # Confirmation: require the previous N‑1 periods also have positive momentum
        prior_mom = close.shift(1).iloc[-self.confirm_periods] / close.shift(1).iloc[-self.confirm_periods - self.lookback] - 1.0
        if pd.isna(prior_mom) or prior_mom <= 0:
            return None

        # Trend filter: price above its moving average
        ma = close.rolling(self.ma_period).mean().iloc[-1]
        if pd.isna(ma) or close.iloc[-1] <= ma:
            return None

        # Volume filter (optional – only applied if volume column exists)
        if "volume" in data.columns:
            vol_avg = data["volume"].rolling(self.vol_lookback).mean().iloc[-1]
            if pd.isna(vol_avg) or data["volume"].iloc[-1] <= self.vol_min_factor * vol_avg:
                return None

        # Confidence scaling – base 0.6 plus a fraction of excess momentum
        excess = max(0.0, mom - self.mom_threshold)
        confidence = min(0.95, 0.60 + excess * 10)  # cap at 0.95

        metadata = {
            "momentum_pct": round(float(mom) * 100, 2),
            "lookback": self.lookback,
            "ma": round(float(ma), 4),
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

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Back‑test logic mirroring the live entry/exit rules."""
        close = df["close"]
        # Momentum series (lagged to avoid look‑ahead)
        prior = close.shift(1)
        mom = prior / prior.shift(self.lookback) - 1.0

        # Basic momentum filter
        mom_ok = mom > self.mom_threshold

        # Confirmation: previous period also positive momentum
        prior_mom = prior.shift(1) / prior.shift(1).shift(self.lookback) - 1.0
        confirm_ok = prior_mom > 0

        # Trend filter: price above moving average
        ma = close.rolling(self.ma_period).mean()
        trend_ok = close > ma

        # Volume filter (if present)
        if "volume" in df.columns:
            vol_avg = df["volume"].rolling(self.vol_lookback).mean()
            vol_ok = df["volume"] > self.vol_min_factor * vol_avg
        else:
            vol_ok = pd.Series(True, index=df.index)

        entries = mom_ok & confirm_ok & trend_ok & vol_ok
        # Exit when momentum falls below threshold or price drops below MA
        exits = (mom <= self.mom_threshold) | (close <= ma)

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
        )