"""
Low Volatility Factor Strategy — Baker, Bradley, Wurgler (2011).

Buy low-volatility stocks (bottom 20% rolling 252-day std).
Historically achieves higher Sharpe than market with lower drawdown.

In single-symbol mode: score the symbol vs a universe, signal when it's
in the low-vol regime and trending up.
"""
import pandas as pd
import numpy as np
from typing import Optional

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class LowVolatilityStrategy(AbstractStrategy):
    name = "low_volatility"
    display_name = "Low Volatility Factor"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    DEFAULT_PARAMS = {
        "lookback_days": 252,
        "top_pct": 30,
        "bottom_pct": 20,
        "rebalance_freq": 21,
    }

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.vol_period = effective["lookback_days"]
        self.vol_percentile = effective["top_pct"]
        self.rebalance_freq = effective["rebalance_freq"]
        self.trend_ema = params.get("trend_ema", 50) if params else 50

    async def analyze(self, data: Optional[pd.DataFrame], symbol: str) -> Optional[Signal]:
        # Guard against None or malformed input
        if not isinstance(data, pd.DataFrame) or data.empty:
            return None
        if "close" not in data.columns or data["close"].empty:
            return None

        # Ensure sufficient history for volatility calculation
        if len(data) < self.vol_period + 10:
            return None

        close = data["close"]
        daily_returns = close.pct_change()
        rolling_vol = daily_returns.rolling(self.vol_period).std() * np.sqrt(252)

        # Current volatility may be NaN if insufficient data; handle gracefully
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else np.nan
        if pd.isna(current_vol):
            return None

        ema50 = close.ewm(span=self.trend_ema).mean().iloc[-1]
        price = close.iloc[-1]

        # Historical vol distribution for ranking
        historical_vols = rolling_vol.dropna()
        if len(historical_vols) < 10:
            return None

        # Percentile rank: avoid division by zero
        percentile_rank = (historical_vols < current_vol).mean() * 100

        if percentile_rank <= self.vol_percentile and price > ema50:
            # Confidence scaling capped at 0.80
            confidence = min(
                0.80,
                0.55 + (self.vol_percentile - percentile_rank) / self.vol_percentile * 0.3,
            )
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "annualized_vol": round(float(current_vol), 4),
                    "vol_percentile": round(percentile_rank, 1),
                },
            )
        return None

    def backtest_signals(self, df: Optional[pd.DataFrame]) -> BacktestSignals:
        # Defensive checks for None or missing data
        if not isinstance(df, pd.DataFrame) or df.empty:
            # Return empty boolean Series matching no index
            empty_series = pd.Series(dtype=bool)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        if "close" not in df.columns or df["close"].empty:
            empty_series = pd.Series(dtype=bool, index=df.index)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        close = df["close"]
        daily_ret = close.pct_change()
        rolling_vol = daily_ret.rolling(self.vol_period).std().shift(1) * np.sqrt(252)
        ema = close.ewm(span=self.trend_ema).mean().shift(1)
        close_s = close.shift(1)

        # Guard against cases where rolling_vol is all NaN
        if rolling_vol.isna().all():
            entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)
            return BacktestSignals(entries=entries, exits=exits)

        # Low vol = below configured percentile of its own rolling vol history
        expanding_pct = rolling_vol.expanding().rank(pct=True) * 100

        entries = (expanding_pct <= self.vol_percentile) & (close_s > ema)
        exits = (expanding_pct > 50) | (close_s < ema)

        # Fill any NaN values to ensure boolean Series
        entries_filled = entries.fillna(False)
        exits_filled = exits.fillna(False)

        return BacktestSignals(entries=entries_filled, exits=exits_filled)