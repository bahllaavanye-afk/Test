from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_FAPI_BASE = "https://fapi.binance.com"

_SYMBOL_MAP: dict[str, str] = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "AVAX/USD": "AVAXUSDT",
    "BNB/USD": "BNBUSDT",
    "XRP/USD": "XRPUSDT",
    "DOGE/USD": "DOGEUSDT",
    "ADA/USD": "ADAUSDT",
}


def _binance_get(path: str, params: dict) -> Any:
    """Fetch JSON from Binance REST endpoint."""
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_FAPI_BASE}{path}?{qs}"
    with urllib.request.urlopen(url, timeout=8) as resp:
        return json.loads(resp.read())


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Use exponential moving average for smoothing
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


class OnChainExchangeNetflowStrategy(AbstractStrategy):
    name = "on_chain_exchange_netflow"
    display_name = "On-Chain Exchange Netflow (OI Proxy, Binance)"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    # --- Configurable thresholds ---
    OI_LOOKBACK = 8  # days of OI history to pull
    OI_THRESHOLD = 0.02  # minimum absolute OI change (2%)
    PRICE_THRESHOLD = 0.01  # minimum absolute price change (1%)
    SMA_PERIOD = 20  # SMA period for trend confirmation
    RSI_PERIOD = 14  # RSI period for overbought/oversold filter
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    CONFIDENCE_BASE = 0.63
    CONFIDENCE_SCALE = 0.13

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate a trading signal based on OI and price dynamics."""
        binance_sym = _SYMBOL_MAP.get(symbol)
        if binance_sym is None:
            return None
        if len(data) < max(self.SMA_PERIOD, self.OI_LOOKBACK) or "close" not in data.columns:
            return None

        # Fetch OI history (blocking I/O moved to thread pool)
        try:
            raw_oi = await asyncio.to_thread(
                _binance_get,
                "/futures/data/openInterestHist",
                {"symbol": binance_sym, "period": "1d", "limit": self.OI_LOOKBACK + 2},
            )
        except Exception as e:
            print(f"    ⚠ OI fetch failed for {binance_sym}: {e}", flush=True)
            return None

        if not raw_oi or len(raw_oi) < 4:
            return None

        oi_values = [float(r.get("sumOpenInterest", 0)) for r in raw_oi]
        if any(v <= 0 for v in oi_values[-4:]):
            return None

        # OI change over the last 3 days
        oi_change_3d = oi_values[-1] / oi_values[-4] - 1.0

        close = data["close"].astype(float)
        price_change_3d = close.iloc[-1] / close.iloc[-4] - 1.0

        # Determine direction signs
        oi_sign = (
            1
            if oi_change_3d > self.OI_THRESHOLD
            else -1
            if oi_change_3d < -self.OI_THRESHOLD
            else 0
        )
        price_sign = (
            1
            if price_change_3d > self.PRICE_THRESHOLD
            else -1
            if price_change_3d < -self.PRICE_THRESHOLD
            else 0
        )
        if oi_sign == 0 or price_sign == 0:
            return None

        # Netflow direction (positive → bullish, negative → bearish)
        netflow = oi_sign * price_sign
        side = "buy" if netflow > 0 else "sell"

        # --- Confirmation filters ---
        # 1. Trend confirmation via SMA
        sma = close.rolling(self.SMA_PERIOD).mean()
        if pd.isna(sma.iloc[-1]):
            return None
        price_above_sma = close.iloc[-1] > sma.iloc[-1]
        if (side == "buy" and not price_above_sma) or (side == "sell" and price_above_sma):
            return None

        # 2. RSI filter to avoid extreme overbought/oversold conditions
        rsi_series = _rsi(close, period=self.RSI_PERIOD)
        rsi_latest = rsi_series.iloc[-1]
        if side == "buy" and rsi_latest > self.RSI_OVERBOUGHT:
            return None
        if side == "sell" and rsi_latest < self.RSI_OVERSOLD:
            return None

        # --- Confidence calculation ---
        oi_mag = min(abs(oi_change_3d) / 0.10, 1.0)
        price_mag = min(abs(price_change_3d) / 0.05, 1.0)
        confidence = min(
            self.CONFIDENCE_BASE + (oi_mag + price_mag) * self.CONFIDENCE_SCALE,
            0.89,
        )
        if confidence < self.confidence_threshold:
            return None

        spot = float(close.iloc[-1])
        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=spot,
            metadata={
                "oi_change_3d": round(oi_change_3d, 4),
                "price_change_3d": round(price_change_3d, 4),
                "oi_sign": oi_sign,
                "price_sign": price_sign,
                "binance_sym": binance_sym,
                "sma": round(sma.iloc[-1], 2),
                "rsi": round(rsi_latest, 2),
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Generate back‑testable entry/exit signals using only price data."""
        if "close" not in df.columns or len(df) < max(self.SMA_PERIOD, 20):
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
                short_entries=pd.Series(False, index=df.index),
            )

        close = df["close"].astype(float)
        price_change_3d = close.pct_change(3)
        sma = close.rolling(self.SMA_PERIOD).mean()
        rsi_series = _rsi(close, period=self.RSI_PERIOD)

        # Entry filter: price momentum + trend + RSI
        bullish_entry = (
            (price_change_3d.shift(1) > self.PRICE_THRESHOLD)
            & (close.shift(1) > sma.shift(1))
            & (rsi_series.shift(1) < self.RSI_OVERBOUGHT)
        )
        bearish_entry = (
            (price_change_3d.shift(1) < -self.PRICE_THRESHOLD)
            & (close.shift(1) < sma.shift(1))
            & (rsi_series.shift(1) > self.RSI_OVERSOLD)
        )

        # Exit filter: price stagnation or reversal
        exit_condition = (
            price_change_3d.shift(1).abs() < 0.005
        ) | (
            (price_change_3d.shift(1) > self.PRICE_THRESHOLD) & (close.shift(1) < sma.shift(1))
        ) | (
            (price_change_3d.shift(1) < -self.PRICE_THRESHOLD) & (close.shift(1) > sma.shift(1))
        )

        entries = bullish_entry.fillna(False)
        short_entries = bearish_entry.fillna(False)
        exits = exit_condition.fillna(False)

        return BacktestSignals(
            entries=entries,
            short_entries=short_entries,
            exits=exits,
        )