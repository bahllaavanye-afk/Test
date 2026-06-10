"""
Dollar Carry Strategy
======================
Academic basis:
  - Carry trade in currencies: borrow low-yield currency (USD when strong), invest in
    high-yield currencies / EM assets.  Classic FX literature: Lustig, Roussanov &
    Verdelhan (2011) "Common Risk Factors in Currency Markets", RFS.
  - DXY weakness is historically correlated with EM equity outperformance: when USD
    weakens, dollar-denominated EM earnings look better, capital flows to EM, and
    commodity tailwinds add fuel.

ETF Implementation (yfinance-tradeable):
  UUP  — Invesco DB US Dollar Index Bullish Fund (DXY proxy, long dollar)
  EEM  — iShares MSCI Emerging Markets ETF (EM equities basket)

Signal construction:
  DXY trend:  UUP SMA20 vs SMA50 (simple moving average crossover)
  Downtrend (UUP SMA20 < SMA50) → long EEM (carry trade on).
  Uptrend   (UUP SMA20 > SMA50) → flat / reduce EM exposure (carry trade off).

  Confidence scaled by magnitude of SMA gap (larger gap = stronger trend).
  Stop: exit if UUP crosses above its 50-day SMA.

Documented Sharpe: 0.6-1.0 (Lustig et al. 2011, EM carry strategies)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DOLLAR_ETF = "UUP"
_EM_ETF     = "EEM"

_SMA_SHORT = 20
_SMA_LONG  = 50
_LOOKBACK  = 252


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


class DollarCarryStrategy(AbstractStrategy):
    """
    Dollar carry: long high-yield EM equities when DXY weakens, short/reduce when strengthens.
    Uses UUP (dollar ETF), EEM (EM equities) via yfinance.
    Entry: 20-day DXY downtrend (UUP SMA20 below SMA50) → long EEM.
    Exit: 20-day DXY uptrend → reduce.
    Risk bucket: directional, market_type: equity
    """

    name = "dollar_carry"
    display_name = "Dollar Carry (UUP/EEM)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.sma_short = int(p.get("sma_short", _SMA_SHORT))
        self.sma_long  = int(p.get("sma_long",  _SMA_LONG))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        uup = _fetch_yf(_DOLLAR_ETF)
        eem = _fetch_yf(_EM_ETF)
        if uup is None or eem is None:
            return None
        if len(uup) < self.sma_long + 5:
            return None

        sma20 = float(uup.rolling(self.sma_short).mean().iloc[-1])
        sma50 = float(uup.rolling(self.sma_long).mean().iloc[-1])

        if np.isnan(sma20) or np.isnan(sma50):
            return None

        gap_pct = (sma20 - sma50) / max(sma50, 1e-8)
        eem_price = float(eem.iloc[-1])

        # DXY downtrend → long EM carry
        if sma20 < sma50:
            confidence = min(0.90, 0.60 + abs(gap_pct) * 5.0)
            return Signal(
                symbol=_EM_ETF if symbol not in (_EM_ETF, _DOLLAR_ETF) else symbol,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=eem_price,
                metadata={
                    "dollar_trend": "down",
                    "uup_sma20": round(sma20, 4),
                    "uup_sma50": round(sma50, 4),
                    "gap_pct": round(gap_pct, 4),
                    "academic_ref": "Lustig, Roussanov & Verdelhan (2011) RFS",
                },
            )

        # DXY uptrend → reduce / exit EM
        if sma20 > sma50:
            confidence = min(0.85, 0.60 + abs(gap_pct) * 5.0)
            return Signal(
                symbol=_EM_ETF if symbol not in (_EM_ETF, _DOLLAR_ETF) else symbol,
                side="sell",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=eem_price,
                metadata={
                    "dollar_trend": "up",
                    "uup_sma20": round(sma20, 4),
                    "uup_sma50": round(sma50, 4),
                    "gap_pct": round(gap_pct, 4),
                    "academic_ref": "Lustig, Roussanov & Verdelhan (2011) RFS",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.sma_long + 5:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # If df contains UUP price use it; else use df["close"] as proxy
        if "uup_close" in df.columns:
            uup_price = df["uup_close"].astype(float)
        else:
            uup_price = df["close"].astype(float)

        sma_s = uup_price.rolling(self.sma_short).mean()
        sma_l = uup_price.rolling(self.sma_long).mean()

        # Dollar downtrend → long EM
        raw_entries = (sma_s < sma_l)
        raw_exits   = (sma_s >= sma_l)

        entries = raw_entries.shift(1).fillna(False).astype(bool)
        exits   = raw_exits.shift(1).fillna(False).astype(bool)

        # Short leg: dollar uptrend
        short_entries = raw_exits.shift(1).fillna(False).astype(bool)
        short_exits   = raw_entries.shift(1).fillna(False).astype(bool)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
