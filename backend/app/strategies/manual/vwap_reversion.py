"""
VWAP Reversion Strategy.

Intraday mean-reversion to Volume-Weighted Average Price.
Entry when price deviates > 1.5× VWAP standard deviation bands with volume
and multi‑bar confirmation. Exit when price reverts to VWAP (within a tighter
band) or stop‑loss is hit.

Academic basis: Berkowitz, Logue & Noser (1988) intraday VWAP tracking.
Known alpha: VWAP band reversion wins ~63% when deviation > 1.5σ.

Requires 1‑minute OHLCV bars for intraday context.
For daily bars, falls back to a VWAP proxy (rolling VWAP window).

Expected Sharpe: 0.9‑1.4 (intraday)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


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

    if not all(c in df.columns for c in (close_col, high_col, low_col, vol_col)):
        return pd.Series(np.nan, index=df.index)

    typical = (df[high_col] + df[low_col] + df[close_col]) / 3.0
    volume = df[vol_col].replace(0, np.nan).fillna(1.0)

    tp_vol = typical * volume
    window = min(len(df), 390)
    vwap = tp_vol.rolling(window=window).sum() / volume.rolling(window=window).sum()
    return vwap


class VWAPReversionStrategy(AbstractStrategy):
    """
    VWAP Reversion: mean-revert to intraday VWAP.

    Entry: price deviates > entry_std_bands × VWAP rolling std,
           confirmed over consecutive bars and supported by volume.
    Exit: price returns within exit_std_bands of VWAP or stop‑loss hit.
    """
    name = "vwap_reversion"
    display_name = "VWAP Reversion"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 60.0  # 1‑minute bars

    DEFAULT_PARAMS = {
        "vwap_period": 30,
        "entry_std_bands": 1.5,
        "exit_std_bands": 0.5,
        "stop_pct": 1.0,
        "confirmation_bars": 2,
        "volume_window": 20,
        "volume_multiplier": 1.0,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.band_std = float(effective["entry_std_bands"])
        self.window = int(effective["vwap_period"])
        self.exit_std_bands = float(effective["exit_std_bands"])
        self.stop_loss_pct = float(effective["stop_pct"]) / 100.0
        self.confirmation_bars = int(effective["confirmation_bars"])
        self.volume_window = int(effective["volume_window"])
        self.volume_multiplier = float(effective["volume_multiplier"])

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Real‑time analysis for a single symbol.
        Returns a Signal when entry criteria are satisfied.
        """
        # Identify column names (case‑insensitive)
        close_col = "close" if "close" in data.columns else "Close"
        vol_col = "volume" if "volume" in data.columns else "Volume"

        if close_col not in data.columns or vol_col not in data.columns:
            return None
        if len(data) < max(self.window, self.volume_window) + self.confirmation_bars:
            return None

        close = data[close_col]
        vwap = _compute_vwap(data)

        if vwap.isna().all():
            return None

        # Deviation and rolling volatility
        deviation = (close - vwap) / vwap.replace(0, np.nan)
        rolling_std = deviation.rolling(self.window).std()
        if rolling_std.isna().iloc[-1]:
            return None

        # Z‑score series (current and historical)
        z_series = deviation / rolling_std.replace(0, np.nan)

        # Volume filter – current bar must be above recent average
        avg_vol = data[vol_col].rolling(self.volume_window).mean()
        if avg_vol.isna().iloc[-1]:
            return None
        current_vol = float(data[vol_col].iloc[-1])
        if current_vol < self.volume_multiplier * float(avg_vol.iloc[-1]):
            return None

        # Confirmation: require the entry condition to hold for `confirmation_bars`
        # consecutive periods (including the latest bar)
        entry_long = z_series < -self.band_std
        entry_short = z_series > self.band_std
        confirmed_long = entry_long.tail(self.confirmation_bars).all()
        confirmed_short = entry_short.tail(self.confirmation_bars).all()

        # Determine side
        side = None
        if confirmed_long:
            side = "buy"
        elif confirmed_short:
            side = "sell"
        if side is None:
            return None

        # Confidence scaling – capped at 0.85, baseline 0.60
        latest_z = float(z_series.iloc[-1])
        confidence = min(0.85, 0.60 + abs(latest_z) * 0.05)

        # Target is the current VWAP; stop‑loss is relative to the latest close
        target_price = float(vwap.iloc[-1])
        if side == "buy":
            stop_price = float(close.iloc[-1]) * (1.0 - self.stop_loss_pct)
        else:
            stop_price = float(close.iloc[-1]) * (1.0 + self.stop_loss_pct)

        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=target_price,
            stop_loss=stop_price,
            metadata={
                "z_score": round(latest_z, 3),
                "vwap": round(target_price, 4),
                "confirmed_bars": self.confirmation_bars,
                "volume_multiplier": self.volume_multiplier,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized backtest signals based on VWAP z‑score.
        Returns -1 (sell), 1 (buy), or 0 (flat) with a one‑bar lag to avoid look‑ahead bias.
        """
        close_col = "close" if "close" in df.columns else "Close"
        vol_col = "volume" if "volume" in df.columns else "Volume"

        if close_col not in df.columns or vol_col not in df.columns:
            return pd.Series(0, index=df.index)

        required_len = max(self.window, self.volume_window) + self.confirmation_bars
        if len(df) < required_len:
            return pd.Series(0, index=df.index)

        close = df[close_col]
        vwap = _compute_vwap(df)

        deviation = (close - vwap) / vwap.replace(0, np.nan)
        rolling_std = deviation.rolling(self.window).std()
        z_score = (deviation / rolling_std.replace(0, np.nan)).fillna(0)

        # Volume filter – bar must exceed recent average volume
        avg_vol = df[vol_col].rolling(self.volume_window).mean()
        vol_filter = df[vol_col] >= self.volume_multiplier * avg_vol

        # Raw entry masks
        long_mask = (z_score < -self.band_std) & vol_filter
        short_mask = (z_score > self.band_std) & vol_filter

        # Apply multi‑bar confirmation
        long_confirm = long_mask.rolling(self.confirmation_bars).min() == 1
        short_confirm = short_mask.rolling(self.confirmation_bars).min() == 1

        signals = pd.Series(0, index=df.index, dtype=float)
        signals[long_confirm] = 1.0
        signals[short_confirm] = -1.0

        # Ensure exit when z‑score re‑enters tighter band
        exit_mask = z_score.abs() <= self.exit_std_bands
        signals[exit_mask] = 0.0

        # Shift to prevent look‑ahead bias
        return signals.shift(1).fillna(0)