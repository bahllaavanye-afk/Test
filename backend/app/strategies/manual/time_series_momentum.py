"""
Time Series Momentum (TSMOM)
============================
Academic basis:
  - Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum" Journal of
    Financial Economics 104(2). The seminal paper documenting that an
    asset's own past 12-month return predicts its next 1-month return,
    *independent* of the cross-section. Documented across 58 instruments
    in equities, commodities, FX, and bonds with Sharpe ≈ 1.2 unlevered.
  - Asness, Chandra, Ilmanen, Israel (2017) "Trend Filtering Reduces
    Hedge Fund Tail Risk" — extension showing TSMOM provides crisis alpha.
  - Hurst, Ooi, Pedersen (2017) "A Century of Evidence on Trend-Following
    Investing" JPM — confirms TSMOM works back to 1880.

Distinction from cross-sectional momentum:
  Cross-sectional (Jegadeesh-Titman 1993): rank assets, long winners short
  losers within a universe at the same time.
  Time-series (this strategy): each asset evaluated *only against its own
  past*. If its 12-month return is positive → long; negative → short.
  Sized inversely to realized volatility for constant ex-ante risk.

Signal:
  excess_return_12m = (close / close.shift(252) - 1)
  position_sign     = sign(excess_return_12m)
  vol_scalar        = target_vol / realized_vol_60d   (capped at 3×)
  position_size     = position_sign × vol_scalar

  Long  if excess_return_12m > 0 by a meaningful margin and passes
        multi‑layer confirmation filters.
  Short if excess_return_12m < 0 by a meaningful margin and passes
        confirmations.
  Exit on sign flip, diminishing momentum, or adverse volatility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class TimeSeriesMomentumStrategy(AbstractStrategy):
    name = "time_series_momentum"
    display_name = "Time Series Momentum (Moskowitz-Ooi-Pedersen)"
    market_type = "equity"  # works on equities, futures, FX, commodities — equity for Alpaca
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86_400  # daily rebalance per the paper

    LOOKBACK = 252           # 12 trading months
    TARGET_VOL = 0.40        # 40% annualized vol target per asset (paper uses 40%)
    VOL_LOOKBACK = 60        # 3-month realized vol
    ENTRY_THRESHOLD = 0.02   # min |12m return| of 2% to take a position
    EXIT_THRESHOLD = 0.005   # exit when |12m return| drops below 0.5%
    CONFIRM_LOOKBACK = 63    # 3‑month return for confirmation
    SMA_PERIOD = 20          # short‑term trend filter
    MAX_VOL = 0.60           # upper bound on realized vol for entry

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.lookback = int(p.get("lookback", self.LOOKBACK))
        self.target_vol = float(p.get("target_vol", self.TARGET_VOL))
        self.vol_lookback = int(p.get("vol_lookback", self.VOL_LOOKBACK))
        self.entry_threshold = float(p.get("entry_threshold", self.ENTRY_THRESHOLD))
        self.exit_threshold = float(p.get("exit_threshold", self.EXIT_THRESHOLD))
        self.confirm_lookback = int(p.get("confirm_lookback", self.CONFIRM_LOOKBACK))
        self.sma_period = int(p.get("sma_period", self.SMA_PERIOD))
        self.max_vol = float(p.get("max_vol", self.MAX_VOL))

    def description(self) -> str:
        return (
            "Each asset's own 12‑month excess return predicts next‑month return. "
            "Position size is inversely proportional to realized volatility. "
            "Additional confirmation filters tighten entry signals. "
            "Source: Moskowitz, Ooi & Pedersen JFE 2012."
        )

    def _compute_filters(self, df: pd.DataFrame) -> dict:
        """Compute auxiliary series used for entry/exit decisions."""
        close = df["close"].astype(float)

        # 12‑month excess return
        excess_ret = (close / close.shift(self.lookback)) - 1

        # 3‑month return for confirmation
        confirm_ret = (close / close.shift(self.confirm_lookback)) - 1

        # 20‑day simple moving average (price trend filter)
        sma = close.rolling(self.sma_period, min_periods=self.sma_period).mean()

        # Realized volatility (annualized) based on log returns
        log_ret = np.log(close / close.shift(1))
        realized_vol = (
            log_ret.rolling(self.vol_lookback, min_periods=20).std() * np.sqrt(252)
        )

        return {
            "excess_ret": excess_ret,
            "confirm_ret": confirm_ret,
            "sma": sma,
            "realized_vol": realized_vol,
        }

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        false_series = pd.Series(False, index=df.index)

        if "close" not in df.columns or len(df) < self.lookback + 10:
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        filters = self._compute_filters(df)

        # Shift to avoid look‑ahead bias: yesterday's calculations drive today's orders
        excess_prev = filters["excess_ret"].shift(1)
        confirm_prev = filters["confirm_ret"].shift(1)
        sma_prev = filters["sma"].shift(1)
        vol_prev = filters["realized_vol"].shift(1)

        # Core momentum condition
        long_core = excess_prev > self.entry_threshold
        short_core = excess_prev < -self.entry_threshold

        # Confirmation: 3‑month return must be aligned with 12‑month signal
        long_confirm = confirm_prev > 0
        short_confirm = confirm_prev < 0

        # Price trend filter: price above SMA for longs, below for shorts
        long_trend = df["close"] > sma_prev
        short_trend = df["close"] < sma_prev

        # Volatility filter: only enter when volatility is below max_vol
        vol_filter = vol_prev < self.max_vol

        entries = (
            long_core
            & long_confirm
            & long_trend
            & vol_filter
        ).fillna(False).astype(bool)

        short_entries = (
            short_core
            & short_confirm
            & short_trend
            & vol_filter
        ).fillna(False).astype(bool)

        # Exit when momentum weakens (below half entry threshold) or sign flips,
        # and also when volatility spikes above max_vol.
        exit_condition = (
            excess_prev.abs() < max(self.exit_threshold, self.entry_threshold / 2)
        ) | (vol_prev > self.max_vol)

        exits = exit_condition.fillna(True).astype(bool)
        short_exits = exit_condition.fillna(True).astype(bool)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in df.columns or len(df) < self.lookback + 5:
            return None

        close = df["close"].astype(float)

        # Compute required series
        filters = self._compute_filters(df)

        # Current (most recent) values
        ret_12m = float(filters["excess_ret"].iloc[-1])
        confirm_ret = float(filters["confirm_ret"].iloc[-1])
        sma = float(filters["sma"].iloc[-1])
        rv = float(filters["realized_vol"].iloc[-1])

        # Basic sanity checks
        if np.isnan(ret_12m) or np.isnan(confirm_ret) or np.isnan(sma) or np.isnan(rv):
            return None

        if abs(ret_12m) < self.entry_threshold:
            return None

        # Confirmation filters
        if (ret_12m > 0 and confirm_ret <= 0) or (ret_12m < 0 and confirm_ret >= 0):
            return None
        if (ret_12m > 0 and close.iloc[-1] <= sma) or (ret_12m < 0 and close.iloc[-1] >= sma):
            return None
        if rv > self.max_vol:
            return None

        # Position sizing scalar
        vol_scalar = min(self.target_vol / max(rv, 0.05), 3.0)

        side = "buy" if ret_12m > 0 else "sell"

        # Confidence is higher for stronger momentum, capped at 0.95
        momentum_strength = min(abs(ret_12m), 0.50)
        confidence = float(min(0.95, 0.45 + momentum_strength * 0.9))

        # ATR‑based stop distance as a volatility‑aware safeguard
        atr_proxy = (
            df["high"].astype(float) - df["low"].astype(float)
        ).rolling(14, min_periods=5).mean().iloc[-1]
        if np.isnan(atr_proxy):
            atr_proxy = close.iloc[-1] * 0.02  # fallback 2% of price

        stop_distance = float(atr_proxy) * 3.0
        last_price = float(close.iloc[-1])

        return Signal(
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            symbol=symbol,
            side=side,
            confidence=confidence,
            target_price=last_price,
            stop_loss=last_price - stop_distance if side == "buy" else last_price + stop_distance,
            take_profit=None,  # hold until sign flip or adverse conditions
            metadata={
                "order_type": "market",
                "ret_12m": round(ret_12m, 4),
                "confirm_ret_3m": round(confirm_ret, 4),
                "realized_vol_60d": round(rv, 4),
                "vol_scalar": round(vol_scalar, 3),
                "sma_20d": round(sma, 4),
            },
        )