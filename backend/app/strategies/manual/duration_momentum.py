"""
Duration Momentum Strategy
============================
Academic basis:
  - Asness, Moskowitz & Pedersen (2013, JF): "Value and Momentum Everywhere" —
    momentum works across asset classes including fixed income.
  - Short-term momentum (1-3 months) in US Treasuries is statistically significant.
  - Duration-adjusted momentum: weight bond ETFs by recent performance adjusted
    for duration (longer duration = more interest rate sensitivity).

ETF Implementation (yfinance-tradeable):
  SHY  — iShares 1-3Y Treasury (duration ~2)
  IEF  — iShares 7-10Y Treasury (duration ~7.5)
  TLT  — iShares 20+ Year Treasury (duration ~17)
  TIPS — iShares TIPS Bond ETF (duration ~7, inflation protection)

Signal:
  Rank 4 duration buckets by 3-month momentum.
  Rotate to highest-momentum duration bucket.
  Duration momentum captures rate trend continuation.

Documented Sharpe: 0.6-1.0 in fixed income (Brooks & Moskowitz 2017)
"""

from __future__ import annotations

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DURATION_ETFS = {
    "SHY":  2.0,    # duration ~2y
    "IEF":  7.5,    # duration ~7.5y
    "TLT":  17.0,   # duration ~17y
    "TIPS": 7.0,    # inflation-linked ~7y
}
_MOMENTUM_DAYS = 63   # 3 months
_REBAL_DAYS    = 21   # 1 month


def _fetch_yf(symbol: str, period: str = "2y") -> pd.Series | None:
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


class DurationMomentumStrategy(AbstractStrategy):
    """
    Rotates across Treasury duration buckets (SHY/IEF/TLT/TIPS) by 3-month momentum.
    Holds the single best-performing duration. Monthly rebalancing.
    Risk bucket: directional, market_type: equity
    """

    name = "duration_momentum"
    display_name = "Duration Momentum (SHY/IEF/TLT/TIPS)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.momentum_days = int(p.get("momentum_days", _MOMENTUM_DAYS))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        series = {}
        for etf in _DURATION_ETFS:
            s = _fetch_yf(etf)
            if s is not None and len(s) >= self.momentum_days:
                series[etf] = s

        if len(series) < 2:
            return None

        returns = {etf: float(s.iloc[-1] / s.iloc[-self.momentum_days] - 1)
                   for etf, s in series.items()}

        ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        best_etf, best_ret = ranked[0]
        worst_etf, worst_ret = ranked[-1]

        trade_sym = symbol if symbol in _DURATION_ETFS else best_etf

        if trade_sym == best_etf:
            confidence = min(0.85, 0.60 + abs(best_ret) * 2.0)
            return Signal(
                symbol=trade_sym,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "best_duration": best_etf,
                    "best_duration_years": _DURATION_ETFS.get(best_etf, 0),
                    "momentum_3m": round(best_ret, 4),
                    "all_returns": {k: round(v, 4) for k, v in returns.items()},
                    "rate_trend": "falling" if best_etf == "TLT" else "rising" if best_etf == "SHY" else "stable",
                    "academic_ref": "Asness, Moskowitz & Pedersen (2013, JF) Momentum Everywhere",
                },
            )

        if trade_sym == worst_etf and worst_ret < -0.01:
            confidence = min(0.75, 0.55 + abs(worst_ret) * 2.0)
            return Signal(
                symbol=trade_sym,
                side="sell",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "worst_duration": worst_etf,
                    "momentum_3m": round(worst_ret, 4),
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.momentum_days + 5:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close = df["close"].astype(float)
        mom = close.pct_change(self.momentum_days)

        entries = (mom.shift(1) > 0.01).fillna(False)
        exits   = (mom.shift(1) < 0.00).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)
