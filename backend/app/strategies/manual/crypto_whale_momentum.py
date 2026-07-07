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

# --- Constants ---
DEFAULT_SYMBOL = "BTCUSDT"
USDT_SUFFIX = "USDT"
INTERVAL = "1h"
LIMIT = 25
BINANCE_KLINES_URL = (
    f"https://api.binance.com/api/v3/klines"
    f"?symbol={{symbol}}&interval={INTERVAL}&limit={LIMIT}"
)
KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
]

TICK_INTERVAL_SECONDS = 3600.0  # hourly

DEFAULT_SPIKE_MULTIPLIER = 3.0
DEFAULT_LOOKBACK_HOURS = 24

CONFIDENCE_MAX = 0.85
CONFIDENCE_BASE = 0.60
CONFIDENCE_FACTOR = 0.05

SIDE_BUY = "buy"
SIDE_SELL = "sell"

META_VOLUME_RATIO = "volume_ratio"
META_PRICE_CHANGE_PCT = "price_change_pct"


def _fetch_binance_klines(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame | None:
    """Fetch last 25 hourly klines from Binance public REST."""
    try:
        url = BINANCE_KLINES_URL.format(symbol=symbol)
        with urllib.request.urlopen(url, timeout=5) as resp:
            raw = json.loads(resp.read())
        df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
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
    tick_interval_seconds = TICK_INTERVAL_SECONDS

    SPIKE_MULTIPLIER = DEFAULT_SPIKE_MULTIPLIER
    LOOKBACK_HOURS = DEFAULT_LOOKBACK_HOURS

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
        if not binance_symbol.endswith(USDT_SUFFIX):
            binance_symbol = DEFAULT_SYMBOL

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
            side = SIDE_BUY if price_change > 0 else SIDE_SELL

            confidence = min(
                CONFIDENCE_MAX,
                CONFIDENCE_BASE + (vol_ratio - self.spike_multiplier) * CONFIDENCE_FACTOR,
            )
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    META_VOLUME_RATIO: round(vol_ratio, 2),
                    META_PRICE_CHANGE_PCT: round(price_change * 100, 3),
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
            vol_ratio = (
                ret.abs()
                / ret.abs().rolling(self.lookback_hours).mean()
            ).shift(1)
            ret_s = ret.shift(1)

        spike = vol_ratio > self.spike_multiplier

        entries = spike & (ret_s > 0)       # spike + up move
        exits = ~spike
        short_entries = spike & (ret_s < 0)  # spike + down move
        short_exits = ~spike

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )