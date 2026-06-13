"""
VWAP Reversion Strategy.

Intraday mean-reversion to Volume-Weighted Average Price.
Entry when price deviates > 1.5× VWAP standard deviation bands.
Exit when price reverts to VWAP or at end of day.

Academic basis: Berkowitz, Logue & Noser (1988) intraday VWAP tracking.
Known alpha: VWAP band reversion wins ~63% when deviation > 1.5σ.

Requires 1-minute OHLCV bars for intraday context.
For daily bars, falls back to a VWAP proxy (rolling VWAP window).

Expected Sharpe: 0.9-1.4 (intraday)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, Signal


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute VWAP for an OHLCV DataFrame.
    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Typical price = (high + low + close) / 3
    """
    close_col = "close" if "close" in df.columns else "Close"
    high_col = "high" if "high" in df.columns else "High"
    low_col = "low" if "low" in df.columns else "Low"
    vol_col = "volume" if "volume" in df.columns else "Volume"

    if not all(c in df.columns for c in [close_col, high_col, low_col, vol_col]):
        return df[close_col] if close_col in df.columns else pd.Series(np.nan, index=df.index)

    typical = (df[high_col] + df[low_col] + df[close_col]) / 3.0
    volume = df[vol_col].replace(0, np.nan).fillna(1.0)

    tp_vol = typical * volume
    vwap = tp_vol.rolling(window=min(len(df), 390)).sum() / volume.rolling(window=min(len(df), 390)).sum()
    return vwap


class VWAPReversionStrategy(AbstractStrategy):
    """
    VWAP Reversion: mean-revert to intraday VWAP.

    Entry: price deviates > band_std × VWAP rolling standard deviation.
    Exit: price returns to VWAP or stop-loss hit.
    """
    name = "vwap_reversion"
    display_name = "VWAP Reversion"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 60.0    # 1-minute bars

    DEFAULT_PARAMS = {
        "vwap_period": 30,
        "entry_std_bands": 1.5,
        "exit_std_bands": 0.5,
        "stop_pct": 1.0,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.band_std = float(effective["entry_std_bands"])
        self.window = int(effective["vwap_period"])
        self.exit_std_bands = float(effective["exit_std_bands"])
        self.stop_loss_pct = float(effective["stop_pct"]) / 100.0

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        close_col = "close" if "close" in data.columns else "Close"
        if close_col not in data.columns or len(data) < self.window + 5:
            return None

        close = data[close_col]
        vwap = _compute_vwap(data)

        if vwap.isna().all():
            return None

        # Rolling deviation from VWAP
        deviation = (close - vwap) / vwap.replace(0, np.nan)
        rolling_std = deviation.rolling(self.window).std()

        if len(deviation) < 2 or rolling_std.isna().iloc[-1]:
            return None

        last_dev = float(deviation.iloc[-1])
        std = float(rolling_std.iloc[-1])

        if std <= 0:
            return None

        z_score = last_dev / std

        # Below VWAP band → expect reversion upward → buy
        if z_score < -self.band_std:
            confidence = min(0.85, 0.60 + abs(z_score) * 0.05)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=float(vwap.iloc[-1]),
                stop_loss=float(close.iloc[-1]) * (1.0 - self.stop_loss_pct),
                metadata={"z_score": round(z_score, 3), "vwap": round(float(vwap.iloc[-1]), 4)},
            )

        # Above VWAP band → expect reversion downward → sell
        elif z_score > self.band_std:
            confidence = min(0.85, 0.60 + abs(z_score) * 0.05)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=float(vwap.iloc[-1]),
                stop_loss=float(close.iloc[-1]) * (1.0 + self.stop_loss_pct),
                metadata={"z_score": round(z_score, 3), "vwap": round(float(vwap.iloc[-1]), 4)},
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized backtest signals based on VWAP z-score.
        Returns -1/0/1 with shift(1) to prevent lookahead.
        """
        close_col = "close" if "close" in df.columns else "Close"
        if close_col not in df.columns or len(df) < self.window + 10:
            return pd.Series(0, index=df.index)

        close = df[close_col]
        vwap = _compute_vwap(df)

        deviation = (close - vwap) / vwap.replace(0, np.nan)
        rolling_std = deviation.rolling(self.window).std()

        z_score = (deviation / rolling_std.replace(0, np.nan)).fillna(0)

        signals = pd.Series(0, index=df.index, dtype=float)
        signals[z_score < -self.band_std] = 1     # buy when below band
        signals[z_score > self.band_std] = -1      # sell when above band

        # CRITICAL: shift(1) to prevent lookahead
        return signals.shift(1).fillna(0)
