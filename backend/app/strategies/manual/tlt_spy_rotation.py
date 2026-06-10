"""
TLT/SPY Risk-Parity Rotation
===============================
Academic basis:
  - Faber (2007): "A Quantitative Approach to Tactical Asset Allocation", SSRN.
  - Simple rules-based equity/bond rotation using 10-month moving averages.
  - When SPY is above its 200-day MA → equity allocation.
  - When TLT is above its 200-day MA but SPY is below → bond allocation.
  - Reduces drawdown significantly vs buy-and-hold SPY.

  Volatility-adjusted version:
  - Allocate to TLT when realized 20-day vol of SPY > 20% annualised.
  - This acts as automatic defensive switch during equity stress.

ETF Implementation (yfinance-tradeable):
  SPY — S&P 500 ETF
  TLT — 20+ Year Treasury Bond ETF
  GLD — Gold (included as third allocation in stressed regime)

Documented Sharpe: 0.8-1.0, max DD < 15% (Faber 2007 backtest 1973-2006)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_SPY = "SPY"
_TLT = "TLT"
_GLD = "GLD"

_MA_WINDOW       = 200
_VOL_WINDOW      = 20
_VOL_THRESHOLD   = 0.20   # annualised vol threshold — above this = stressed


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


class TLTSPYRotationStrategy(AbstractStrategy):
    """
    Rotates between SPY (equities) and TLT (long bonds) based on 200-day MA signals.
    Adds GLD allocation when equity vol > 20% annualised.
    Risk bucket: directional, market_type: equity
    """

    name = "tlt_spy_rotation"
    display_name = "TLT/SPY Risk-Parity Rotation (Faber)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.ma_window     = int(p.get("ma_window",     _MA_WINDOW))
        self.vol_window    = int(p.get("vol_window",    _VOL_WINDOW))
        self.vol_threshold = float(p.get("vol_threshold", _VOL_THRESHOLD))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        spy = _fetch_yf(_SPY)
        tlt = _fetch_yf(_TLT)
        if spy is None or tlt is None or len(spy) < self.ma_window:
            return None

        spy_ma = float(spy.rolling(self.ma_window).mean().iloc[-1])
        tlt_ma = float(tlt.rolling(self.ma_window).mean().iloc[-1])
        spy_now = float(spy.iloc[-1])
        tlt_now = float(tlt.iloc[-1])

        spy_rvol = float(spy.pct_change().rolling(self.vol_window).std().iloc[-1]) * np.sqrt(252)
        high_vol = spy_rvol > self.vol_threshold

        trade_sym = symbol if symbol in (_SPY, _TLT, _GLD) else _SPY

        if spy_now > spy_ma and not high_vol:
            confidence = min(0.85, 0.65 + (spy_now / spy_ma - 1) * 3.0)
            return Signal(
                symbol=_SPY if trade_sym == _SPY else trade_sym,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "regime": "equity",
                    "spy_above_200ma": True,
                    "spy_rvol": round(spy_rvol, 4),
                    "academic_ref": "Faber (2007) Tactical Asset Allocation",
                },
            )

        if tlt_now > tlt_ma or high_vol:
            trade = _TLT if not high_vol else (_GLD if trade_sym == _GLD else _TLT)
            confidence = min(0.82, 0.62 + (spy_rvol - self.vol_threshold) * 1.5 if high_vol else 0.72)
            return Signal(
                symbol=trade,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "regime": "bonds" if not high_vol else "stressed_bonds",
                    "spy_above_200ma": spy_now > spy_ma,
                    "tlt_above_200ma": tlt_now > tlt_ma,
                    "spy_rvol": round(spy_rvol, 4),
                    "high_vol_regime": high_vol,
                    "academic_ref": "Faber (2007) TAA",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.ma_window + 5:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close   = df["close"].astype(float)
        ma      = close.rolling(self.ma_window).mean()
        rvol    = close.pct_change().rolling(self.vol_window).std() * np.sqrt(252)

        spy_above = close.shift(1) > ma.shift(1)
        low_vol   = rvol.shift(1) < self.vol_threshold

        entries = (spy_above & low_vol).fillna(False)
        exits   = (~spy_above | ~low_vol).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)
