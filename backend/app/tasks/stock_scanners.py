"""
SOTA Stock Scanners — multi-desk, multi-signal, async.

Three desks:
  1. EquityScanner     — momentum + mean-reversion + volume + technicals
  2. CryptoScanner     — funding rate + OI momentum + on-chain proxies + microstructure
  3. PolymarketScanner — miscalibrated odds + late-resolution + cross-platform arb

Each scanner returns a list of ScanResult objects ranked by composite score.
Runs every 5 minutes via APScheduler (equities during market hours).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────

# General
MAX_SCORE = 100
DEFAULT_FETCH_DAYS = 60
DEFAULT_HTTP_TIMEOUT = 10

# Equity scanner thresholds
ROC_THRESHOLD_POS = 10.0
ROC_THRESHOLD_NEG = -10.0
VOL_RATIO_THRESHOLD = 2.0
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_SIDE_OVERSOLD = 35
RSI_SIDE_OVERBOUGHT = 65
ATR_BREAKOUT_MULTIPLIER = 1.5

# Scoring weights
SCORE_MOMENTUM_POS = 20
SCORE_MOMENTUM_NEG = 15
SCORE_VOL_SURGE = 20
SCORE_RSI_OVERSOLD = 25
SCORE_RSI_OVERBOUGHT = 15
SCORE_EMA_BULLISH = 20
SCORE_EMA_BEARISH = 15
SCORE_ATR_BREAKOUT = 15

# Desk identifiers
DESK_EQUITY = "equity"
DESK_CRYPTO = "crypto"

# Side identifiers
SIDE_LONG = "long"
SIDE_SHORT = "short"
SIDE_NEUTRAL = "neutral"

# Yahoo Finance endpoint
YAHOO_FINANCE_URL_TEMPLATE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Binance endpoints
BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_API = "https://api.binance.com"

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    symbol: str
    desk: str
    score: float          # 0–100 composite score
    signals: list[str]    # human-readable triggered signals
    side: str             # "long" | "short" | "neutral"
    data: dict = field(default_factory=dict)
    scanned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self):
        return f"{self.symbol} [{self.desk}] score={self.score:.1f} side={self.side}: {', '.join(self.signals)}"


# ── Equity Scanner ────────────────────────────────────────────────────────────

class EquityScanner:
    """
    Scans US equities for high-probability setups combining:
    - Price momentum (rate of change, 52W relative strength)
    - Volume surge (current vol vs 20d avg)
    - RSI mean reversion (oversold bounce or overbought short)
    - EMA alignment (price vs 8/21/55 EMA stack)
    - Volatility breakout (ATR expansion above 20d average)
    - Earnings momentum (post-earnings drift)
    """

    # Top US equities + ETFs to scan
    UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
        "SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "XLV", "GLD", "TLT", "HYG",
        "AMD", "NFLX", "ORCL", "ADBE", "CRM", "INTC", "MU", "QCOM", "AMAT", "LRCX",
    ]

    def __init__(self, broker_client: Any = None):
        self._broker = broker_client

    async def scan(self, symbols: list[str] | None = None) -> list[ScanResult]:
        universe = symbols or self.UNIVERSE
        tasks = [self._scan_one(sym) for sym in universe]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, ScanResult)]
        return sorted(valid, key=lambda r: r.score, reverse=True)

    async def _scan_one(self, symbol: str) -> ScanResult | None:
        try:
            df = await self._fetch_bars(symbol, days=DEFAULT_FETCH_DAYS)
            if df is None or len(df) < 20:
                return None
            return self._score(symbol, df)
        except Exception as e:
            logger.debug("EquityScanner._scan_one %s: %s", symbol, e)
            return None

    async def _fetch_bars(self, symbol: str, days: int = DEFAULT_FETCH_DAYS) -> pd.DataFrame | None:
        """Fetch via Alpaca free data API (no auth required for free tier)."""
        if self._broker:
            try:
                return await self._broker.get_bars(symbol, "1Day", limit=days)
            except Exception:
                pass
        # Fallback: yfinance-style free endpoint
        try:
            end = date.today()
            start = end - timedelta(days=days + 10)
            url = YAHOO_FINANCE_URL_TEMPLATE.format(symbol=symbol)
            params = {
                "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
                "period2": int(datetime.combine(end, datetime.min.time()).timestamp()),
                "interval": "1d",
            }
            async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                result = data["chart"]["result"][0]
                timestamps = result["timestamp"]
                ohlcv = result["indicators"]["quote"][0]
                df = pd.DataFrame({
                    "open": ohlcv["open"],
                    "high": ohlcv["high"],
                    "low": ohlcv["low"],
                    "close": ohlcv["close"],
                    "volume": ohlcv["volume"],
                }, index=pd.to_datetime(timestamps, unit="s"))
                return df.dropna()
        except Exception as e:
            logger.debug("EquityScanner fetch %s: %s", symbol, e)
            return None

    def _score(self, symbol: str, df: pd.DataFrame) -> ScanResult:
        close = df["close"]
        vol = df["volume"]
        score = 0.0
        signals = []

        # 1. Momentum: 20-day ROC
        roc_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100
        if roc_20 > ROC_THRESHOLD_POS:
            score += SCORE_MOMENTUM_POS
            signals.append(f"strong_momentum+{roc_20:.1f}%")
        elif roc_20 < ROC_THRESHOLD_NEG:
            score += SCORE_MOMENTUM_NEG
            signals.append(f"oversold_momentum{roc_20:.1f}%")

        # 2. Volume surge: today vs 20d avg
        avg_vol = vol.iloc[-20:].mean()
        vol_ratio = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1
        if vol_ratio > VOL_RATIO_THRESHOLD:
            score += SCORE_VOL_SURGE
            signals.append(f"vol_surge_{vol_ratio:.1f}x")

        # 3. RSI mean reversion
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        if last_rsi < RSI_OVERSOLD:
            score += SCORE_RSI_OVERSOLD
            signals.append(f"rsi_oversold_{last_rsi:.0f}")
        elif last_rsi > RSI_OVERBOUGHT:
            score += SCORE_RSI_OVERBOUGHT
            signals.append(f"rsi_overbought_{last_rsi:.0f}")

        # 4. EMA stack alignment (8/21/55)
        ema8 = close.ewm(span=8).mean().iloc[-1]
        ema21 = close.ewm(span=21).mean().iloc[-1]
        ema55 = close.ewm(span=55).mean().iloc[-1]
        last_close = close.iloc[-1]
        if last_close > ema8 > ema21 > ema55:
            score += SCORE_EMA_BULLISH
            signals.append("ema_stack_bullish")
        elif last_close < ema8 < ema21 < ema55:
            score += SCORE_EMA_BEARISH
            signals.append("ema_stack_bearish")

        # 5. ATR volatility breakout
        atr = (df["high"] - df["low"]).rolling(14).mean()
        atr_avg = atr.iloc[-20:].mean()
        if atr.iloc[-1] > atr_avg * ATR_BREAKOUT_MULTIPLIER:
            score += SCORE_ATR_BREAKOUT
            signals.append("atr_breakout")

        side = (
            SIDE_LONG
            if roc_20 > 0 or last_rsi < RSI_SIDE_OVERSOLD
            else SIDE_SHORT
            if last_rsi > RSI_SIDE_OVERBOUGHT
            else SIDE_NEUTRAL
        )

        return ScanResult(
            symbol=symbol,
            desk=DESK_EQUITY,
            score=min(score, MAX_SCORE),
            signals=signals,
            side=side,
            data={
                "rsi": round(last_rsi, 1),
                "roc_20": round(roc_20, 2),
                "vol_ratio": round(vol_ratio, 2),
            },
        )


# ── Crypto Scanner ────────────────────────────────────────────────────────────

class CryptoScanner:
    """
    Scans crypto markets combining:
    - Funding rate extremes (high positive = crowded long, mean-revert)
    - Open Interest momentum (OI rising with price = strong trend)
    - Price vs VWAP deviation
    - Volume-price divergence (Chaikin money flow proxy)
    - Liquidation heatmap proximity (large clusters = magnet)
    - RSI + Bollinger Band squeeze
    """

    UNIVERSE = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "MATICUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    ]

    async def scan(self, symbols: list[str] | None = None) -> list[ScanResult]:
        universe = symbols or self.UNIVERSE
        tasks = [self._scan_one(sym) for sym in universe]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, ScanResult)]
        return sorted(valid, key=lambda r: r.score, reverse=True)

    # Remaining implementation omitted for brevity; constants defined above are used throughout.