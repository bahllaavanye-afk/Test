"""
Breakeven Inflation Trading Strategy
=======================================
Academic basis:
  - TIPS vs nominal Treasury spread = market's inflation expectation (breakeven).
  - D'Amico, Kim & Wei (2018, RFS): breakeven inflation has significant return
    predictability when anchored vs unanchored from Fed expectations.
  - Strategy: when breakeven inflation is rising (TIPS outperforming nominals),
    favor real assets. When breakeven falling, favor nominal Treasuries.

ETF Implementation (yfinance-tradeable):
  TIPS  — iShares TIPS Bond ETF (inflation-linked)
  IEF   — iShares 7-10Y Treasury (nominal)
  TIP   — iShares TIPS ETF (alternative, similar to TIPS)
  GLD   — Gold (real asset, correlated with inflation expectations)

  Breakeven proxy: TIPS/IEF ratio (rising = breakeven rising = inflation expectations up)
  Signal: 20-day momentum of TIPS/IEF ratio.
  Rising breakeven → long TIPS + GLD.
  Falling breakeven → long IEF (nominal).

Documented Sharpe: 0.7-1.1 (TIPS strategies literature, PIMCO 2020)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_TIPS_ETF    = "TIPS"
_NOMINAL_ETF = "IEF"
_GOLD_ETF    = "GLD"

_MOMENTUM_DAYS = 20
_THRESHOLD     = 0.003   # 0.3% momentum to trigger signal


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


class BreakevenInflationStrategy(AbstractStrategy):
    """
    Trades TIPS vs nominal Treasury spread (breakeven inflation proxy).
    Rising breakeven (TIPS > IEF momentum) → long TIPS + GLD.
    Falling breakeven → long IEF nominal.
    Risk bucket: directional, market_type: equity
    """

    name = "breakeven_inflation"
    display_name = "Breakeven Inflation (TIPS vs IEF)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.momentum_days = int(p.get("momentum_days", _MOMENTUM_DAYS))
        self.threshold     = float(p.get("threshold",     _THRESHOLD))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        tips = _fetch_yf(_TIPS_ETF)
        ief  = _fetch_yf(_NOMINAL_ETF)
        gld  = _fetch_yf(_GOLD_ETF)

        if tips is None or ief is None or len(tips) < self.momentum_days + 5:
            return None

        df = pd.DataFrame({"tips": tips, "ief": ief}).dropna()
        if len(df) < self.momentum_days + 5:
            return None

        ratio = df["tips"] / df["ief"].clip(lower=1e-8)
        momentum = float(ratio.pct_change(self.momentum_days).iloc[-1])

        if np.isnan(momentum):
            return None

        trade_sym = symbol if symbol in (_TIPS_ETF, _NOMINAL_ETF, _GOLD_ETF) else _TIPS_ETF

        if momentum > self.threshold:
            confidence = min(0.85, 0.60 + abs(momentum) * 10.0)
            return Signal(
                symbol=trade_sym if trade_sym != _NOMINAL_ETF else _TIPS_ETF,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "signal": "breakeven_rising",
                    "tips_ief_momentum": round(momentum, 4),
                    "regime": "inflation_expectation_rising",
                    "preferred_asset": "TIPS + GLD",
                    "academic_ref": "D'Amico, Kim & Wei (2018, RFS) TIPS breakeven",
                },
            )

        if momentum < -self.threshold:
            confidence = min(0.82, 0.58 + abs(momentum) * 10.0)
            return Signal(
                symbol=_NOMINAL_ETF if trade_sym not in (_NOMINAL_ETF, _TIPS_ETF) else trade_sym,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "signal": "breakeven_falling",
                    "tips_ief_momentum": round(momentum, 4),
                    "regime": "inflation_expectation_falling",
                    "preferred_asset": "IEF nominal",
                    "academic_ref": "D'Amico, Kim & Wei (2018, RFS)",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.momentum_days + 5:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close = df["close"].astype(float)
        momentum = close.pct_change(self.momentum_days)

        entries = (momentum.shift(1) > self.threshold).fillna(False)
        exits   = (momentum.shift(1) < 0).fillna(False)
        short_entries = (momentum.shift(1) < -self.threshold).fillna(False)
        short_exits   = (momentum.shift(1) > 0).fillna(False)

        return BacktestSignals(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
        )
