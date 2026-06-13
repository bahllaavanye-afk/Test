"""
Yield Spread Mean Reversion Strategy
=======================================
Academic basis:
  - Credit spreads (IG and HY) exhibit strong mean-reversion to long-run averages.
    Duffee (1998, JF): corporate bond spread mean reversion.
  - When HY spreads spike above historical norms, they tend to compress back.
    Long HYG (high yield ETF) when spreads are above 2-std rolling mean.
    Short (or hold TLT) when spreads are compressed below 2-std rolling mean.

ETF Implementation (yfinance-tradeable):
  HYG  — iShares HY Corp Bond ETF (price inversely related to HY spread)
  LQD  — iShares IG Corp Bond ETF
  IEF  — iShares 7-10Y Treasury ETF (duration hedge)

  Spread proxy: HYG/IEF ratio (HY minus treasury return proxy)
  When ratio dips below lower Bollinger Band → spread spike → buy HYG.
  When ratio rises above upper Bollinger Band → spread compressed → sell HYG.

Sharpe target: 0.8-1.2 (bond carry + mean reversion)
"""

from __future__ import annotations

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_HYG = "HYG"
_IEF = "IEF"
_BB_WINDOW = 60
_BB_WIDTH   = 2.0


def _fetch_yf(symbol: str, period: str = "3y") -> pd.Series | None:
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist.empty:
            return None
        closes = hist["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes
    except Exception:
        return None


class YieldSpreadReversionStrategy(AbstractStrategy):
    """
    Mean reverts HY spread (via HYG/IEF ratio) to its Bollinger Band midpoint.
    Wide spread (ratio below lower band) → buy HYG. Tight spread → reduce.
    Risk bucket: arbitrage, market_type: equity
    """

    name = "yield_spread_reversion"
    display_name = "Yield Spread Mean Reversion (HYG/IEF)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 86400.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.bb_window = int(p.get("bb_window", _BB_WINDOW))
        self.bb_width  = float(p.get("bb_width",  _BB_WIDTH))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        hyg = _fetch_yf(_HYG)
        ief = _fetch_yf(_IEF)
        if hyg is None or ief is None:
            return None

        df = pd.DataFrame({"hyg": hyg, "ief": ief}).dropna()
        if len(df) < self.bb_window + 5:
            return None

        ratio = df["hyg"] / df["ief"].clip(lower=1e-8)
        ma = ratio.rolling(self.bb_window).mean()
        std = ratio.rolling(self.bb_window).std()
        upper = ma + self.bb_width * std
        lower = ma - self.bb_width * std

        now_ratio = float(ratio.iloc[-1])
        now_lower = float(lower.iloc[-1])
        now_upper = float(upper.iloc[-1])
        now_ma    = float(ma.iloc[-1])

        trade_sym = symbol if symbol in (_HYG, _IEF) else _HYG

        if now_ratio < now_lower:
            dev = (now_lower - now_ratio) / max(std.iloc[-1], 1e-8)
            confidence = min(0.88, 0.65 + dev * 0.08)
            return Signal(
                symbol=trade_sym,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "signal": "spread_wide",
                    "hyg_ief_ratio": round(now_ratio, 4),
                    "lower_band": round(now_lower, 4),
                    "std_devs_from_ma": round(dev, 2),
                    "academic_ref": "Duffee (1998, JF) credit spread mean reversion",
                },
            )

        if now_ratio > now_upper:
            dev = (now_ratio - now_upper) / max(std.iloc[-1], 1e-8)
            confidence = min(0.85, 0.60 + dev * 0.08)
            return Signal(
                symbol=trade_sym,
                side="sell",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "signal": "spread_tight",
                    "hyg_ief_ratio": round(now_ratio, 4),
                    "upper_band": round(now_upper, 4),
                    "std_devs_from_ma": round(dev, 2),
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.bb_window + 5:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close = df["close"].astype(float)
        ma    = close.rolling(self.bb_window).mean()
        std   = close.rolling(self.bb_window).std()
        lower = ma - self.bb_width * std
        upper = ma + self.bb_width * std

        c_shift = close.shift(1)
        entries = (c_shift < lower.shift(1)).fillna(False)
        exits   = (c_shift > ma.shift(1)).fillna(False)
        short_entries = (c_shift > upper.shift(1)).fillna(False)
        short_exits   = (c_shift < ma.shift(1)).fillna(False)

        return BacktestSignals(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
        )
