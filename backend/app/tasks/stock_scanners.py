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
            df = await self._fetch_bars(symbol, days=60)
            if df is None or len(df) < 20:
                return None
            return self._score(symbol, df)
        except Exception as e:
            logger.debug("EquityScanner._scan_one %s: %s", symbol, e)
            return None

    async def _fetch_bars(self, symbol: str, days: int = 60) -> pd.DataFrame | None:
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
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
                "period2": int(datetime.combine(end, datetime.min.time()).timestamp()),
                "interval": "1d",
            }
            async with httpx.AsyncClient(timeout=10) as client:
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
        if roc_20 > 10:
            score += 20
            signals.append(f"strong_momentum+{roc_20:.1f}%")
        elif roc_20 < -10:
            score += 15
            signals.append(f"oversold_momentum{roc_20:.1f}%")

        # 2. Volume surge: today vs 20d avg
        avg_vol = vol.iloc[-20:].mean()
        vol_ratio = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1
        if vol_ratio > 2.0:
            score += 20
            signals.append(f"vol_surge_{vol_ratio:.1f}x")

        # 3. RSI mean reversion
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        if last_rsi < 30:
            score += 25
            signals.append(f"rsi_oversold_{last_rsi:.0f}")
        elif last_rsi > 70:
            score += 15
            signals.append(f"rsi_overbought_{last_rsi:.0f}")

        # 4. EMA stack alignment (8/21/55)
        ema8 = close.ewm(span=8).mean().iloc[-1]
        ema21 = close.ewm(span=21).mean().iloc[-1]
        ema55 = close.ewm(span=55).mean().iloc[-1]
        last_close = close.iloc[-1]
        if last_close > ema8 > ema21 > ema55:
            score += 20
            signals.append("ema_stack_bullish")
        elif last_close < ema8 < ema21 < ema55:
            score += 15
            signals.append("ema_stack_bearish")

        # 5. ATR volatility breakout
        atr = (df["high"] - df["low"]).rolling(14).mean()
        atr_avg = atr.iloc[-20:].mean()
        if atr.iloc[-1] > atr_avg * 1.5:
            score += 15
            signals.append("atr_breakout")

        side = "long" if roc_20 > 0 or last_rsi < 35 else "short" if last_rsi > 65 else "neutral"

        return ScanResult(
            symbol=symbol, desk="equity", score=min(score, 100), signals=signals, side=side,
            data={"rsi": round(last_rsi, 1), "roc_20": round(roc_20, 2), "vol_ratio": round(vol_ratio, 2)},
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
    BINANCE_FAPI = "https://fapi.binance.com"
    BINANCE_API = "https://api.binance.com"

    async def scan(self, symbols: list[str] | None = None) -> list[ScanResult]:
        universe = symbols or self.UNIVERSE
        tasks = [self._scan_one(sym) for sym in universe]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, ScanResult)]
        return sorted(valid, key=lambda r: r.score, reverse=True)

    async def _scan_one(self, symbol: str) -> ScanResult | None:
        try:
            klines, funding = await asyncio.gather(
                self._fetch_klines(symbol),
                self._fetch_funding_rate(symbol),
                return_exceptions=True,
            )
            if isinstance(klines, Exception) or klines is None or len(klines) < 20:
                return None
            return self._score(symbol, klines, funding if not isinstance(funding, Exception) else None)
        except Exception as e:
            logger.debug("CryptoScanner._scan_one %s: %s", symbol, e)
            return None

    async def _fetch_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame | None:
        url = f"{self.BINANCE_API}/api/v3/klines"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
                resp.raise_for_status()
                data = resp.json()
                df = pd.DataFrame(data, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "trades", "tbav", "tqav", "ignore"
                ])
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                df.index = pd.to_datetime(df["open_time"], unit="ms")
                return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.debug("CryptoScanner._fetch_klines %s: %s", symbol, e)
            return None

    async def _fetch_funding_rate(self, symbol: str) -> float | None:
        try:
            url = f"{self.BINANCE_FAPI}/fapi/v1/premiumIndex"
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(url, params={"symbol": symbol})
                resp.raise_for_status()
                data = resp.json()
                return float(data.get("lastFundingRate", 0))
        except Exception:
            return None

    def _score(self, symbol: str, df: pd.DataFrame, funding_rate: float | None) -> ScanResult:
        close = df["close"]
        vol = df["volume"]
        score = 0.0
        signals = []

        # 1. Funding rate signal
        if funding_rate is not None:
            fr_pct = funding_rate * 100
            if fr_pct > 0.05:   # extremely crowded long
                score += 20
                signals.append(f"funding_crowded_long_{fr_pct:.3f}%")
            elif fr_pct < -0.02:
                score += 20
                signals.append(f"funding_crowded_short_{fr_pct:.3f}%")

        # 2. RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))
        last_rsi = rsi.iloc[-1]
        if last_rsi < 28:
            score += 25
            signals.append(f"rsi_extremely_oversold_{last_rsi:.0f}")
        elif last_rsi > 72:
            score += 20
            signals.append(f"rsi_extremely_overbought_{last_rsi:.0f}")

        # 3. Bollinger Band squeeze / breakout
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        last_close = close.iloc[-1]
        bandwidth = (upper - lower) / sma20
        if last_close > upper.iloc[-1]:
            score += 20
            signals.append("bb_breakout_upper")
        elif last_close < lower.iloc[-1]:
            score += 20
            signals.append("bb_breakout_lower")
        if bandwidth.iloc[-1] < bandwidth.iloc[-20:].quantile(0.2):
            score += 15
            signals.append("bb_squeeze")

        # 4. Volume momentum (volume rising with price)
        vol_ma = vol.rolling(20).mean()
        price_roc = close.pct_change(5).iloc[-1]
        vol_roc = (vol.iloc[-1] / vol_ma.iloc[-1] - 1) if vol_ma.iloc[-1] > 0 else 0
        if price_roc > 0.03 and vol_roc > 0.5:
            score += 15
            signals.append("volume_price_momentum")

        side = "long" if last_rsi < 35 or last_close < lower.iloc[-1] else \
               "short" if last_rsi > 65 or last_close > upper.iloc[-1] else "neutral"

        return ScanResult(
            symbol=symbol, desk="crypto", score=min(score, 100), signals=signals, side=side,
            data={"rsi": round(last_rsi, 1), "funding_rate": funding_rate, "bb_width": round(float(bandwidth.iloc[-1]), 4)},
        )


# ── Polymarket Scanner ────────────────────────────────────────────────────────

class PolymarketScanner:
    """
    Scans Polymarket for high-value opportunities:
    - Miscalibrated probability (YES+NO < $0.97 → free money)
    - Late-resolution arbitrage (near-certain outcome, not yet resolved)
    - High-volume markets with momentum
    - Markets with >$50k open interest
    """
    CLOB_API = "https://clob.polymarket.com"

    async def scan(self) -> list[ScanResult]:
        try:
            markets = await self._fetch_markets()
            results = [self._score_market(m) for m in markets if m]
            valid = [r for r in results if r is not None]
            return sorted(valid, key=lambda r: r.score, reverse=True)[:20]
        except Exception as e:
            logger.warning("PolymarketScanner.scan: %s", e)
            return []

    async def _fetch_markets(self) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{self.CLOB_API}/markets",
                    params={"active": "true", "closed": "false", "limit": 100}
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("data", [])
        except Exception as e:
            logger.debug("PolymarketScanner._fetch_markets: %s", e)
            return []

    def _score_market(self, market: dict) -> ScanResult | None:
        try:
            question = market.get("question", "Unknown")
            slug = market.get("market_slug", question[:20])
            tokens = market.get("tokens", [])
            if len(tokens) < 2:
                return None

            yes_price = float(next((t["price"] for t in tokens if t.get("outcome") == "Yes"), 0.5))
            no_price = float(next((t["price"] for t in tokens if t.get("outcome") == "No"), 0.5))
            spread = yes_price + no_price
            vol = float(market.get("volume", 0))

            score = 0.0
            signals = []

            # 1. Binary arb: YES+NO < 0.97
            if spread < 0.97:
                arb_profit = (1 - spread) * 100
                score += min(arb_profit * 20, 50)
                signals.append(f"binary_arb_{arb_profit:.2f}%")

            # 2. Late-resolution: near-certain but not resolved
            if yes_price > 0.92:
                score += 25
                signals.append(f"late_resolution_yes_{yes_price:.2f}")
            elif yes_price < 0.08:
                score += 25
                signals.append(f"late_resolution_no_{no_price:.2f}")

            # 3. High volume / liquidity
            if vol > 50_000:
                score += 15
                signals.append(f"high_vol_${vol:,.0f}")

            # 4. Active market
            if vol > 10_000:
                score += 10
                signals.append("liquid")

            if score < 10:
                return None

            side = "long_yes" if yes_price < 0.5 and spread < 0.97 else \
                   "long_no" if no_price < 0.5 and spread < 0.97 else "neutral"

            return ScanResult(
                symbol=slug, desk="polymarket", score=min(score, 100),
                signals=signals, side=side,
                data={"yes": yes_price, "no": no_price, "spread": round(spread, 4), "volume": vol},
            )
        except Exception:
            return None


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ScannerOrchestrator:
    """Run all three desk scanners and publish results to Redis."""

    def __init__(self, redis_client: Any = None, broker_client: Any = None):
        self._redis = redis_client
        self.equity = EquityScanner(broker_client)
        self.crypto = CryptoScanner()
        self.polymarket = PolymarketScanner()

    async def run_all(self) -> dict[str, list[ScanResult]]:
        equity_results, crypto_results, poly_results = await asyncio.gather(
            self.equity.scan(),
            self.crypto.scan(),
            self.polymarket.scan(),
            return_exceptions=True,
        )
        results = {
            "equity": equity_results if isinstance(equity_results, list) else [],
            "crypto": crypto_results if isinstance(crypto_results, list) else [],
            "polymarket": poly_results if isinstance(poly_results, list) else [],
        }

        if self._redis:
            await self._publish_to_redis(results)

        return results

    async def _publish_to_redis(self, results: dict[str, list[ScanResult]]) -> None:
        import json
        for desk, scans in results.items():
            try:
                payload = json.dumps([{
                    "symbol": r.symbol, "score": r.score,
                    "signals": r.signals, "side": r.side, "data": r.data,
                } for r in scans[:10]])
                await self._redis.set(f"scanner:{desk}:top10", payload, ex=600)
            except Exception as e:
                logger.debug("ScannerOrchestrator._publish_to_redis %s: %s", desk, e)
