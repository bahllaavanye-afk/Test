"""
Yield Curve Momentum Strategy
================================
Academic basis:
  - Harvey (1988) showed the yield curve slope (2Y-10Y spread) predicts US recessions.
  - When the spread steepens (long rates rise relative to short), it signals expansion.
  - When inverted (short > long), it historically precedes equity bear markets.

ETF Implementation (yfinance-tradeable):
  TLT  — iShares 20+ Year Treasury Bond ETF (long-duration proxy, inverse of long rates)
  IEF  — iShares 7-10 Year Treasury Bond ETF (intermediate duration)
  SHY  — iShares 1-3 Year Treasury Bond ETF (short-duration proxy, inverse of short rates)

Curve slope proxy:
  slope_ratio = TLT / SHY
  When TLT rises relative to SHY → long-duration outperforming → curve steepening.
  Rolling z-score of ratio over 252 days gives a normalized spread signal.

Signal construction:
  spread_ratio  = TLT / SHY (long-duration premium; higher = steeper curve)
  slope_mom     = spread_ratio.pct_change(20)  # 20-day momentum
  spread_zscore = rolling z-score of spread_ratio over 252 days

  Steepening + positive z-score → risk-on → long SPY.
  Inversion (z-score < -1) → defensive → reduce equity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_LONG_DURATION  = "TLT"   # 20Y Treasury — inverse of long-end rates
_MID_DURATION   = "IEF"   # 7-10Y Treasury
_SHORT_DURATION = "SHY"   # 1-3Y Treasury — inverse of short-end rates
_EQUITY_ETF     = "SPY"

_LOOKBACK_DAYS  = 252
_MOMENTUM_DAYS  = 20
_ENTRY_Z        = 0.30
_EXIT_Z         = -0.50
_STOP_Z         = -1.20


def _fetch_yf(symbol: str, period: str = "3y") -> pd.Series | None:
    """Fetch adjusted close prices via yfinance. Returns None on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes
    except Exception:
        return None


class YieldCurveMomentumStrategy(AbstractStrategy):
    """
    Trades equity direction based on 2Y-10Y yield curve slope momentum.
    Uses TLT (20Y), IEF (7-10Y), SHY (1-3Y) as rate proxies (all via yfinance).
    Signal: When spread is steepening and above 0 → long SPY.
    When spread is inverting (below 0) → reduce equity / defensive.
    Risk bucket: directional, market_type: equity
    """

    name = "yield_curve_momentum"
    display_name = "Yield Curve Slope Momentum (TLT/IEF/SHY)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.lookback = int(p.get("lookback_days", _LOOKBACK_DAYS))
        self.momentum_days = int(p.get("momentum_days", _MOMENTUM_DAYS))
        self.entry_z = float(p.get("entry_z", _ENTRY_Z))
        self.exit_z = float(p.get("exit_z", _EXIT_Z))
        self.stop_z = float(p.get("stop_z", _STOP_Z))

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window, min_periods=window // 2).mean()
        std = series.rolling(window, min_periods=window // 2).std()
        return (series - mean) / std.clip(lower=1e-8)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        tlt = _fetch_yf(_LONG_DURATION)
        shy = _fetch_yf(_SHORT_DURATION)
        if tlt is None or shy is None:
            return None

        df = pd.DataFrame({"tlt": tlt, "shy": shy}).dropna()
        if len(df) < self.lookback:
            return None

        ratio = df["tlt"] / df["shy"].clip(lower=1e-8)
        zscore = self._rolling_zscore(ratio, self.lookback)
        momentum = ratio.pct_change(self.momentum_days)

        if zscore.empty or np.isnan(zscore.iloc[-1]):
            return None

        z_now = float(zscore.iloc[-1])
        mom_now = float(momentum.iloc[-1]) if not np.isnan(momentum.iloc[-1]) else 0.0
        trade_sym = symbol if symbol in (_EQUITY_ETF, _LONG_DURATION, _SHORT_DURATION) else _EQUITY_ETF

        if z_now > self.entry_z and mom_now > 0:
            confidence = min(0.90, 0.60 + abs(z_now) * 0.10)
            return Signal(
                symbol=trade_sym,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "regime": "curve_steepening",
                    "spread_zscore": round(z_now, 4),
                    "slope_momentum_20d": round(mom_now, 4),
                    "tlt_price": round(float(df["tlt"].iloc[-1]), 2),
                    "shy_price": round(float(df["shy"].iloc[-1]), 2),
                },
            )

        if z_now < self.stop_z:
            confidence = min(0.90, 0.60 + abs(z_now + self.stop_z) * 0.10)
            return Signal(
                symbol=trade_sym,
                side="sell",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "regime": "curve_inverted",
                    "spread_zscore": round(z_now, 4),
                    "slope_momentum_20d": round(mom_now, 4),
                    "tlt_price": round(float(df["tlt"].iloc[-1]), 2),
                    "shy_price": round(float(df["shy"].iloc[-1]), 2),
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.lookback:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close = df["close"].astype(float)

        if "tlt_close" in df.columns and "shy_close" in df.columns:
            tlt_s = df["tlt_close"].astype(float)
            shy_s = df["shy_close"].astype(float)
        else:
            tlt_s = close
            shy_s = close.shift(self.momentum_days).bfill()

        ratio = tlt_s / shy_s.clip(lower=1e-8)
        zscore = self._rolling_zscore(ratio, self.lookback)
        momentum = ratio.pct_change(self.momentum_days)

        z_shift = zscore.shift(1)
        m_shift = momentum.shift(1)

        entries = ((z_shift > self.entry_z) & (m_shift > 0)).fillna(False)
        exits = (z_shift < self.exit_z).fillna(False)
        short_entries = (z_shift < self.stop_z).fillna(False)
        short_exits = (z_shift > self.exit_z).fillna(False)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
