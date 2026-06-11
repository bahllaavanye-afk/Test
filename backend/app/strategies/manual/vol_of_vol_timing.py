"""
Volatility-of-Volatility (VVIX) Regime Timing
Academic basis: Whaley (2013) "Trading Volatility: At What Cost?";
Huang & Shaliastovich (2015) "Volatility-of-Volatility Risk".

VVIX = the 30-day expected volatility of VIX itself. High VVIX signals:
  - Uncertainty about uncertainty → tail risk elevated
  - VIX term structure unstable → mean-reversion strategies unreliable
  - Historically: top-quartile VVIX predicts next-month negative equity returns

This strategy computes a VVIX proxy from publicly available data:
  proxy = rolling std of VIXY (or VXX) daily log-returns over 21 days

Regime signal on any equity/index:
  proxy in bottom-quartile (stable)  → long  (regime favors trend-following)
  proxy in top-quartile (turbulent)  → short (regime favors hedges)

VIXY is an ETF tracking front-month VIX futures; freely available on Alpaca.
"""
from __future__ import annotations

import asyncio
import json
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"

_VIX_PROXY_TICKER = "VIXY"  # iPath Series B VIX Short-Term Futures
_VIX_FALLBACK     = "VXX"   # Barclays iPath VIX Short-Term Futures


def _fetch_closes_sync(ticker: str, days: int) -> pd.Series:
    start = (date.today() - timedelta(days=days + 30)).isoformat()
    try:
        from app.brokers.alpaca_headers import alpaca_headers
        url = (
            f"{_DATA_BASE}/v2/stocks/{ticker}/bars"
            f"?timeframe=1Day&start={start}&limit={days + 30}&adjustment=split&feed=iex"
        )
        req = urllib.request.Request(url, headers=alpaca_headers())
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        bars = data.get("bars", [])
        if not bars:
            return pd.Series(dtype=float)
        s = pd.Series({b["t"]: float(b["c"]) for b in bars})
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)


class VolOfVolTimingStrategy(AbstractStrategy):
    name = "vol_of_vol_timing"
    display_name = "Volatility-of-Volatility Timing (VVIX Proxy)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0
    confidence_threshold = 0.65

    VVIX_WINDOW    = 21    # rolling window for vol-of-vol
    HISTORY_WINDOW = 126   # 6 months for percentile ranking
    LOW_PCTILE     = 25    # below 25th → stable regime → long
    HIGH_PCTILE    = 75    # above 75th → turbulent → short
    MIN_BARS       = 50

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # Fetch VIX proxy
        vix_prices = await asyncio.to_thread(_fetch_closes_sync, _VIX_PROXY_TICKER, self.HISTORY_WINDOW + 30)
        if len(vix_prices) < self.MIN_BARS:
            vix_prices = await asyncio.to_thread(_fetch_closes_sync, _VIX_FALLBACK, self.HISTORY_WINDOW + 30)
        if len(vix_prices) < self.MIN_BARS:
            return None

        vix_log_rets = np.log(vix_prices).diff().dropna()
        vvix_proxy   = vix_log_rets.rolling(self.VVIX_WINDOW, min_periods=10).std().dropna()
        if len(vvix_proxy) < 30:
            return None

        current_vvix = float(vvix_proxy.iloc[-1])
        history      = vvix_proxy.tail(self.HISTORY_WINDOW)
        percentile   = float((history < current_vvix).mean() * 100)

        if percentile <= self.LOW_PCTILE:
            side = "buy"
            conf = min(0.63 + (self.LOW_PCTILE - percentile) / 100.0 * 1.5, 0.88)
        elif percentile >= self.HIGH_PCTILE:
            side = "sell"
            conf = min(0.63 + (percentile - self.HIGH_PCTILE) / 100.0 * 1.5, 0.88)
        else:
            return None

        if conf < self.confidence_threshold:
            return None

        spot = float(data["close"].iloc[-1]) if len(data) > 0 and "close" in data.columns else 0.0
        return Signal(
            symbol=symbol,
            side=side,
            confidence=conf,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=spot,
            metadata={
                "vvix_proxy":  round(current_vvix, 6),
                "percentile":  round(percentile, 1),
                "regime":      "stable" if side == "buy" else "turbulent",
                "vix_ticker":  _VIX_PROXY_TICKER,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.MIN_BARS:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )
        close    = df["close"].astype(float)
        log_rets = np.log(close).diff()

        # Backtest uses asset's own realized vol as VVIX proxy
        vvix_proxy = log_rets.rolling(self.VVIX_WINDOW, min_periods=10).std()
        pctile     = vvix_proxy.rolling(self.HISTORY_WINDOW, min_periods=30).apply(
            lambda w: float((w < w.iloc[-1]).mean() * 100), raw=False
        )
        entries       = (pctile.shift(1) <= self.LOW_PCTILE).fillna(False)
        short_entries = (pctile.shift(1) >= self.HIGH_PCTILE).fillna(False)
        exits         = ((pctile.shift(1) > 40) & (pctile.shift(1) < 60)).fillna(False)
        return BacktestSignals(entries=entries, exits=exits, short_entries=short_entries)
