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

  Long  if excess_return_12m > 0 by a meaningful margin
  Short if excess_return_12m < 0 by a meaningful margin
  Exit on sign flip or |return_12m| < entry_threshold
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

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.lookback = int(p.get("lookback", self.LOOKBACK))
        self.target_vol = float(p.get("target_vol", self.TARGET_VOL))
        self.vol_lookback = int(p.get("vol_lookback", self.VOL_LOOKBACK))
        self.entry_threshold = float(p.get("entry_threshold", self.ENTRY_THRESHOLD))
        self.exit_threshold = float(p.get("exit_threshold", self.EXIT_THRESHOLD))

    def description(self) -> str:
        return (
            "Each asset's own 12-month excess return predicts next-month return. "
            "Sized inversely to realized vol for constant ex-ante risk. "
            "Source: Moskowitz, Ooi & Pedersen JFE 2012."
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        false_series = pd.Series(False, index=df.index)

        if "close" not in df.columns or len(df) < self.lookback + 10:
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))

        # 12-month return
        excess_ret = (close / close.shift(self.lookback)) - 1

        # Realized vol for sizing (kept for metadata; backtest engine handles sizing)
        rv = log_ret.rolling(self.vol_lookback, min_periods=20).std() * np.sqrt(252)

        # shift(1): yesterday's signal determines today's position (no lookahead)
        excess_prev = excess_ret.shift(1)

        entries       = (excess_prev > self.entry_threshold).fillna(False).astype(bool)
        exits         = (excess_prev.abs() < self.exit_threshold).fillna(True).astype(bool)
        short_entries = (excess_prev < -self.entry_threshold).fillna(False).astype(bool)
        short_exits   = (excess_prev.abs() < self.exit_threshold).fillna(True).astype(bool)

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
        last_price = float(close.iloc[-1])
        ret_12m = float(close.iloc[-1] / close.iloc[-(self.lookback + 1)] - 1)

        if abs(ret_12m) < self.entry_threshold:
            return None

        log_ret = np.log(close / close.shift(1))
        rv = float(log_ret.iloc[-self.vol_lookback:].std() * np.sqrt(252))
        if rv < 1e-4 or np.isnan(rv):
            return None

        vol_scalar = min(self.target_vol / max(rv, 0.05), 3.0)
        side = "buy" if ret_12m > 0 else "sell"
        confidence = float(min(0.90, 0.50 + min(abs(ret_12m), 0.50) * 0.8))

        atr_proxy = (df["high"].astype(float) - df["low"].astype(float)).rolling(14).mean().iloc[-1]
        stop_distance = float(atr_proxy) * 3.0 if not np.isnan(atr_proxy) else last_price * 0.05

        return Signal(
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            symbol=symbol,
            side=side,
            confidence=confidence,
            target_price=last_price,
            stop_loss=last_price - stop_distance if side == "buy" else last_price + stop_distance,
            take_profit=None,  # TSMOM holds until sign flip, no fixed TP
            metadata={
                "order_type": "market",
                "ret_12m": round(ret_12m, 4),
                "realized_vol_60d": round(rv, 4),
                "vol_scalar": round(vol_scalar, 3),
            },
        )
