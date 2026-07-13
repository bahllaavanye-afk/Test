"""
On-Chain Exchange Netflow (Crypto)
Academic basis: CryptoQuant research (2020-2023); Gerstein, Yu, Luk (2022)
"Bitcoin Futures-Based ETF vs. Bitcoin: A Comparative Analysis".

Exchange netflow = BTC flowing INTO exchanges (sell pressure) vs OUT (accumulation).
Proxy: change in perpetual futures Open Interest (OI) from Binance public REST.

When OI rises AND price rises → new longs being opened → continuation (buy)
When OI falls AND price falls → liquidations/unwinding → continuation (sell)
When OI rises AND price falls → shorts being opened → bearish
When OI falls AND price rises → shorts being squeezed → bullish

Endpoints (no API key required):
  GET https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1d
"""
from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_FAPI_BASE = "https://fapi.binance.com"

_SYMBOL_MAP: dict[str, str] = {
    "BTC/USD":  "BTCUSDT",
    "ETH/USD":  "ETHUSDT",
    "SOL/USD":  "SOLUSDT",
    "AVAX/USD": "AVAXUSDT",
    "BNB/USD":  "BNBUSDT",
    "XRP/USD":  "XRPUSDT",
    "DOGE/USD": "DOGEUSDT",
    "ADA/USD":  "ADAUSDT",
}


def _binance_get(path: str, params: dict) -> Any:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_FAPI_BASE}{path}?{qs}"
    with urllib.request.urlopen(url, timeout=8) as resp:
        return json.loads(resp.read())


class OnChainExchangeNetflowStrategy(AbstractStrategy):
    name = "on_chain_exchange_netflow"
    display_name = "On-Chain Exchange Netflow (OI Proxy, Binance)"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    OI_LOOKBACK = 8  # days of OI history
    OI_THRESHOLD = 0.02  # 2% OI change required to signal
    PRICE_THRESHOLD = 0.01  # 1% price move required

    async def analyze(self, data: pd.DataFrame | None, symbol: str | None) -> Signal | None:
        # Defensive checks for None or malformed inputs
        if data is None or not isinstance(data, pd.DataFrame):
            return None
        if symbol is None:
            return None

        binance_sym = _SYMBOL_MAP.get(symbol)
        if binance_sym is None:
            return None
        if len(data) < 5 or "close" not in data.columns:
            return None

        # Fetch OI history from Binance (blocking, run in thread pool)
        try:
            raw_oi = await asyncio.to_thread(
                _binance_get,
                "/futures/data/openInterestHist",
                {"symbol": binance_sym, "period": "1d", "limit": self.OI_LOOKBACK + 2},
            )
        except Exception as e:
            print(f"    ⚠ OI fetch failed for {binance_sym}: {e}", flush=True)
            return None

        # Validate OI payload
        if not raw_oi or not isinstance(raw_oi, (list, tuple)):
            return None
        if len(raw_oi) < 4:
            return None

        oi_values = [float(r.get("sumOpenInterest", 0)) for r in raw_oi if isinstance(r, dict)]
        if len(oi_values) < 4:
            return None
        if oi_values[-4] <= 0:
            return None

        oi_change_3d = oi_values[-1] / oi_values[-4] - 1.0

        close = data["close"].astype(float)
        if len(close) < 4:
            return None
        price_change_3d = float(close.iloc[-1] / close.iloc[-4] - 1.0)

        # Signal grid: OI direction × price direction
        oi_sign = (
            1
            if oi_change_3d > self.OI_THRESHOLD
            else (-1 if oi_change_3d < -self.OI_THRESHOLD else 0)
        )
        price_sign = (
            1
            if price_change_3d > self.PRICE_THRESHOLD
            else (-1 if price_change_3d < -self.PRICE_THRESHOLD else 0)
        )

        if oi_sign == 0 or price_sign == 0:
            return None

        netflow = oi_sign * price_sign
        side = "buy" if netflow > 0 else "sell"

        oi_mag = min(abs(oi_change_3d) / 0.10, 1.0)
        price_mag = min(abs(price_change_3d) / 0.05, 1.0)
        conf = min(0.63 + (oi_mag + price_mag) * 0.13, 0.89)

        if conf < self.confidence_threshold:
            return None

        spot = float(close.iloc[-1])
        return Signal(
            symbol=symbol,
            side=side,
            confidence=conf,
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
            },
        )

    def backtest_signals(self, df: pd.DataFrame | None) -> BacktestSignals:
        # Defensive handling for None or malformed DataFrames
        if df is None or not isinstance(df, pd.DataFrame):
            empty_series = pd.Series(False)
            return BacktestSignals(entries=empty_series, exits=empty_series, short_entries=empty_series)

        if "close" not in df.columns or len(df) < 20:
            entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)
            return BacktestSignals(entries=entries, exits=exits, short_entries=entries)

        close = df["close"].astype(float)
        ret3 = close.pct_change(3)
        ret7 = close.pct_change(7)

        # OI unavailable offline; use short + medium momentum agreement as proxy
        entries = ((ret3.shift(1) > self.PRICE_THRESHOLD) & (ret7.shift(1) > 0.02)).fillna(False)
        short_entries = ((ret3.shift(1) < -self.PRICE_THRESHOLD) & (ret7.shift(1) < -0.02)).fillna(False)
        exits = (ret3.shift(1).abs() < 0.005).fillna(False)

        # Ensure the returned series align with the original index even if df is empty
        if df.empty:
            entries = pd.Series(False, index=df.index)
            short_entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)

        return BacktestSignals(entries=entries, exits=exits, short_entries=short_entries)