"""
PMI Sector Rotation Strategy
==============================
Academic basis:
  - The ISM Manufacturing PMI is a leading indicator of economic growth.
    PMI > 50 = expansion; PMI < 50 = contraction.
  - Different equity sectors outperform at different PMI regimes (NBER cycle research).
    PMI > 55 (strong expansion): Industrials (XLI), Materials (XLB), Energy (XLE).
    PMI 50-55 (mild expansion): Technology (XLK), Consumer Discretionary (XLY).
    PMI < 50 (contraction): Utilities (XLU), Consumer Staples (XLP), Healthcare (XLV).

Implementation (ETF proxy via yfinance):
  FRED PMI proxy: ISM PMI is released monthly. We approximate from sector ETF performance:
  when cyclicals outperform defensives, PMI is likely rising.

  Sector momentum ranking: score each sector ETF by 3-month momentum.
  Rotate to top 3 sectors. Rebalance monthly.

Documented Sharpe: 0.6-0.9 (SSGA sector rotation research, 2021)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_CYCLICAL_SECTORS   = ["XLI", "XLB", "XLE", "XLK", "XLY"]
_DEFENSIVE_SECTORS  = ["XLU", "XLP", "XLV", "XLF"]
_ALL_SECTORS        = _CYCLICAL_SECTORS + _DEFENSIVE_SECTORS
_MOMENTUM_DAYS      = 63  # ~3 months
_TOP_N              = 3
_REBAL_DAYS         = 21  # ~1 month rebalancing


def _fetch_yf(symbol: str, period: str = "2y") -> pd.Series | None:
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


class PMISectorRotationStrategy(AbstractStrategy):
    """
    Rotates across 9 SPDR sector ETFs using 3-month momentum as PMI proxy.
    Holds top 3 momentum sectors. Monthly rebalancing.
    Risk bucket: directional, market_type: equity
    """

    name = "pmi_sector_rotation"
    display_name = "PMI Sector Rotation (XLI/XLB/XLK...)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.momentum_days = int(p.get("momentum_days", _MOMENTUM_DAYS))
        self.top_n         = int(p.get("top_n",         _TOP_N))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        series = {}
        for etf in _ALL_SECTORS:
            s = _fetch_yf(etf)
            if s is not None and len(s) >= self.momentum_days:
                series[etf] = s

        if len(series) < 4:
            return None

        returns = {etf: float(s.iloc[-1] / s.iloc[-self.momentum_days] - 1)
                   for etf, s in series.items()}

        ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        top_sectors = [etf for etf, _ in ranked[:self.top_n]]
        bottom_sectors = [etf for etf, _ in ranked[-2:]]

        trade_sym = symbol if symbol in _ALL_SECTORS else top_sectors[0]
        in_top = trade_sym in top_sectors
        in_bottom = trade_sym in bottom_sectors

        cyclical_avg  = np.mean([returns.get(s, 0) for s in _CYCLICAL_SECTORS if s in returns])
        defensive_avg = np.mean([returns.get(s, 0) for s in _DEFENSIVE_SECTORS if s in returns])
        regime = "expansion" if cyclical_avg > defensive_avg else "contraction"

        if in_top:
            best_ret = returns.get(top_sectors[0], 0)
            confidence = min(0.85, 0.60 + abs(best_ret) * 1.0)
            return Signal(
                symbol=trade_sym,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "regime": regime,
                    "top_sectors": top_sectors,
                    "momentum_3m": round(returns.get(trade_sym, 0), 4),
                    "cyclical_vs_defensive": round(cyclical_avg - defensive_avg, 4),
                    "academic_ref": "SSGA sector rotation + ISM PMI cycle research",
                },
            )

        if in_bottom:
            confidence = min(0.75, 0.55 + abs(returns.get(trade_sym, 0)) * 1.0)
            return Signal(
                symbol=trade_sym,
                side="sell",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "regime": regime,
                    "bottom_sectors": bottom_sectors,
                    "momentum_3m": round(returns.get(trade_sym, 0), 4),
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.momentum_days + _REBAL_DAYS:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close = df["close"].astype(float)
        momentum = close.pct_change(self.momentum_days)

        # Signal: buy when momentum > 0 (proxy for being in a top-ranked sector)
        entries = (momentum.shift(1) > 0).fillna(False)
        exits   = (momentum.shift(1) < -0.02).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)
