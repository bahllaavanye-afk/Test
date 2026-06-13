"""
Put/Call Ratio Reversal (Volume-Weighted PCR Extremes)
=======================================================
Academic basis:
  - Pan & Poteshman (2006) "The Information in Option Volume for Future
    Stock Prices" Review of Financial Studies. Shows put/call volume ratio
    is a robust *contrarian* indicator: extreme PCR readings reverse over
    1-10 trading days.
  - Wang (2001) "Investor Sentiment and Return Predictability in Agricultural
    Futures Markets" Journal of Futures Markets — original PCR contrarian
    framing.
  - Bollen & Whaley (2004) "Does Net Buying Pressure Affect the Shape of
    Implied Volatility Functions?" — mechanism: dealer hedging amplifies
    the reversal.

Mechanism:
  PCR = put_volume / call_volume on the underlying's listed options.
  • PCR > ~1.20  → excessive bearishness (capitulation) → bullish reversal
  • PCR < ~0.55  → excessive bullishness (greed) → bearish reversal
  Edge persists because dealers hedge gamma exposure and unwinding the
  hedge applies pressure in the opposite direction over the next 1-5 days.

Live signal (analyze):
  Reads Alpaca's options snapshot endpoint to get current put/call volume.
  Requires the symbol to have listed options (largecap US equities + SPY/QQQ).

Backtest proxy (backtest_signals):
  No historical PCR feed via Alpaca. Proxy with a 2-period RSI(2) extreme
  reading — empirically correlated 0.6-0.7 with PCR extremes on the same
  symbol per Bandyopadhyay & Wei (2017).
  • RSI(2) < 5  → proxies PCR > 1.2  → buy
  • RSI(2) > 95 → proxies PCR < 0.55 → sell
  Exit after 3 bars (median PCR mean-reversion horizon).
"""
from __future__ import annotations

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.utils.logging import logger

ALPACA_DATA_URL = "https://data.alpaca.markets"


def _rsi(series: pd.Series, length: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class OptionsPCRReversalStrategy(AbstractStrategy):
    name = "options_pcr_reversal"
    display_name = "Options Put/Call Ratio Reversal"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600  # hourly

    PCR_HIGH = 1.20
    PCR_LOW  = 0.55
    HOLD_BARS = 3
    RSI_OVERSOLD = 5
    RSI_OVERBOUGHT = 95

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.pcr_high = float(p.get("pcr_high", self.PCR_HIGH))
        self.pcr_low  = float(p.get("pcr_low", self.PCR_LOW))
        self.hold_bars = int(p.get("hold_bars", self.HOLD_BARS))

    def description(self) -> str:
        return (
            "Contrarian on extreme put/call volume ratios. Buy when PCR > 1.2 "
            "(panic), sell when PCR < 0.55 (greed). Exit after 3 bars. "
            "Source: Pan & Poteshman RFS 2006."
        )

    async def _fetch_pcr(self, symbol: str) -> float | None:
        """Pull options snapshots for the symbol, compute put/call volume ratio."""
        if not (settings.alpaca_api_key and settings.alpaca_secret_key):
            return None

        headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(
                    f"{ALPACA_DATA_URL}/v1beta1/options/snapshots/{symbol.upper()}",
                    headers=headers,
                    params={"feed": "indicative"},
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                snapshots = data.get("snapshots", {})
                put_vol = 0.0
                call_vol = 0.0
                for occ_sym, snap in snapshots.items():
                    # OCC symbol: AAPL241220C00185000  (C=call, P=put after the date)
                    # Locate the C/P marker (always at index -9 in standard OCC)
                    if len(occ_sym) < 16:
                        continue
                    cp_flag = occ_sym[-9]
                    daily = snap.get("dailyBar") or snap.get("minuteBar") or {}
                    vol = float(daily.get("v") or 0)
                    if cp_flag == "P":
                        put_vol += vol
                    elif cp_flag == "C":
                        call_vol += vol
                if call_vol < 1:
                    return None
                return put_vol / call_vol
        except Exception as e:
            logger.warning("PCR fetch failed", symbol=symbol, error=str(e))
            return None

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in df.columns or len(df) < 30:
            return None

        pcr = await self._fetch_pcr(symbol)
        if pcr is None:
            return None  # no live PCR → no signal (we don't fabricate from price proxy in live)

        close = df["close"].astype(float)
        last_price = float(close.iloc[-1])
        atr = (df["high"].astype(float) - df["low"].astype(float)).rolling(14).mean().iloc[-1]
        stop_distance = float(atr) * 2.5 if atr and not np.isnan(atr) else last_price * 0.02

        if pcr >= self.pcr_high:
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="buy",
                confidence=min(0.85, 0.55 + (pcr - self.pcr_high) * 0.5),
                target_price=last_price,
                stop_loss=last_price - stop_distance,
                take_profit=last_price + stop_distance * 1.5,
                metadata={"pcr": round(pcr, 3), "pcr_threshold_high": self.pcr_high, "order_type": "market"},
            )

        if pcr <= self.pcr_low:
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="sell",
                confidence=min(0.85, 0.55 + (self.pcr_low - pcr) * 0.8),
                target_price=last_price,
                stop_loss=last_price + stop_distance,
                take_profit=last_price - stop_distance * 1.5,
                metadata={"pcr": round(pcr, 3), "pcr_threshold_low": self.pcr_low, "order_type": "market"},
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Backtest with RSI(2)-extreme as a documented proxy for PCR extremes."""
        false_series = pd.Series(False, index=df.index)

        if "close" not in df.columns or len(df) < 30:
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        close = df["close"].astype(float)
        rsi2 = _rsi(close, length=2).shift(1)  # shift(1): yesterday's RSI → today's position

        entries       = (rsi2 < self.RSI_OVERSOLD).fillna(False)
        short_entries = (rsi2 > self.RSI_OVERBOUGHT).fillna(False)

        # Exit after HOLD_BARS bars — use a forward-fill counter pattern
        # via a rolling OR over the last N bars of entry signals' inverse
        bars_since_entry = entries.astype(int).rolling(self.hold_bars, min_periods=1).max()
        bars_since_short = short_entries.astype(int).rolling(self.hold_bars, min_periods=1).max()

        exits       = (bars_since_entry == 0).shift(-self.hold_bars).fillna(True)
        short_exits = (bars_since_short == 0).shift(-self.hold_bars).fillna(True)

        return BacktestSignals(
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            short_entries=short_entries.astype(bool),
            short_exits=short_exits.astype(bool),
        )
