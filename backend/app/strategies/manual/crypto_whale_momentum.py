"""
Crypto Whale Momentum Strategy.

On-chain/exchange proxy: when hourly BTC volume on Binance spikes > 3× the
24-hour average, a large institutional player ("whale") is likely moving.
Trade in the direction of the spike (momentum, not reversal).

Volume data source: Binance public REST API
  GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=25
"""
import json
import numpy as np
import pandas as pd
import urllib.request
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals

BINANCE_KLINES_URL = (
    "https://api.binance.com/api/v3/klines"
    "?symbol={symbol}&interval=1h&limit=25"
)


def _fetch_binance_klines(symbol: str = "BTCUSDT") -> pd.DataFrame | None:
    """Fetch last 25 hourly klines from Binance public REST."""
    try:
        url = BINANCE_KLINES_URL.format(symbol=symbol)
        with urllib.request.urlopen(url, timeout=5) as resp:
            raw = json.loads(resp.read())
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except Exception:
        return None


class CryptoWhaleMomentumStrategy(AbstractStrategy):
    name = "crypto_whale_momentum"
    display_name = "Crypto Whale Volume Momentum"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0   # hourly

    SPIKE_MULTIPLIER = 3.0   # volume spike threshold vs 24h average
    LOOKBACK_HOURS = 24      # hours for average volume calculation

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.spike_multiplier = p.get("spike_multiplier", self.SPIKE_MULTIPLIER)
        self.lookback_hours = p.get("lookback_hours", self.LOOKBACK_HOURS)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns:
            return None

        # Try live Binance data first
        binance_symbol = symbol.replace("-", "").replace("/", "")
        if not binance_symbol.endswith("USDT"):
            binance_symbol = "BTCUSDT"

        live_df = _fetch_binance_klines(binance_symbol)

        if live_df is not None and len(live_df) >= self.lookback_hours + 1:
            volume = live_df["volume"]
            close_prices = live_df["close"]
        elif "volume" in data.columns and len(data) >= self.lookback_hours + 1:
            volume = data["volume"]
            close_prices = data["close"]
        else:
            return None

        avg_vol = float(volume.iloc[-self.lookback_hours - 1:-1].mean())
        current_vol = float(volume.iloc[-1])
        prev_close = float(close_prices.iloc[-2])
        current_close = float(close_prices.iloc[-1])

        if avg_vol < 1e-8:
            return None

        vol_ratio = current_vol / avg_vol
        price_change = (current_close - prev_close) / prev_close

        if vol_ratio > self.spike_multiplier:
            # Volume spike detected — trade in direction of price move
            if price_change > 0:
                side = "buy"
            else:
                side = "sell"

            confidence = min(0.85, 0.60 + (vol_ratio - self.spike_multiplier) * 0.05)
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "volume_ratio": round(vol_ratio, 2),
                    "price_change_pct": round(price_change * 100, 3),
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        ret = close.pct_change()

        if "volume" in df.columns:
            vol = df["volume"]
            avg_vol = vol.rolling(self.lookback_hours).mean()
            vol_ratio = (vol / avg_vol.replace(0, np.nan)).shift(1)
            ret_s = ret.shift(1)
        else:
            # Fallback: use absolute return as volume proxy
            vol_ratio = (ret.abs() / ret.abs().rolling(self.lookback_hours).mean()).shift(1)
            ret_s = ret.shift(1)

        spike = vol_ratio > self.spike_multiplier

        entries = spike & (ret_s > 0)       # spike + up move
        exits = ~spike
        short_entries = spike & (ret_s < 0) # spike + down move
        short_exits = ~spike

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )
