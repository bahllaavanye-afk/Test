import httpx
import numpy as np
import pandas as pd

from datetime import date, timedelta

from app.config import settings
from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"


class FiftyTwoWeekHighStrategy(AbstractStrategy):
    name = "fifty_two_week_high"
    display_name = "52-Week High Proximity (George & Hwang 2004)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "JPM", "V", "MA", "UNH", "JNJ", "XOM", "HD", "LLY",
        "AVGO", "CRM", "ORCL", "AMD", "COST", "TMO", "AMAT", "MU",
        "PG", "KO", "PEP", "WMT", "DIS", "BAC", "MRK", "CVX",
        "MCD", "ABT", "ACN", "ADBE", "NKE", "CSCO", "AMGN", "GE",
        "HON", "UPS", "CAT", "NFLX", "ABBV",
    ]

    LOOKBACK_DAYS = 252
    NEAR_HIGH_THRESHOLD = 0.95   # Long entry: ratio >= 0.95
    NEAR_HIGH_CEILING = 1.00     # Skip if already AT the new high (blow-off risk)
    BROKEN_THRESHOLD = 0.50      # Skip if too far below
    MIN_DOLLAR_VOLUME = 50_000_000  # $50M minimum daily dollar volume

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    async def _fetch_bars(self, symbol: str, days: int = 260) -> pd.DataFrame:
        start = (date.today() - timedelta(days=days + 30)).isoformat()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "1Day",
                        "start": start,
                        "limit": days + 30,
                        "feed": "iex",
                    },
                    headers=alpaca_headers(),
                )
            if resp.status_code != 200:
                return pd.DataFrame()
            bars = resp.json().get("bars", [])
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(
                [
                    {
                        "t": b["t"],
                        "close": float(b["c"]),
                        "volume": float(b.get("v", 0)),
                    }
                    for b in bars
                ]
            )
            df["t"] = pd.to_datetime(df["t"])
            df = df.set_index("t").sort_index()
            return df
        except Exception:
            return pd.DataFrame()

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if symbol not in self.UNIVERSE:
            return None

        df = await self._fetch_bars(symbol, days=self.LOOKBACK_DAYS + 10)
        if df.empty or len(df) < self.LOOKBACK_DAYS - 30:
            return None

        close = df["close"]
        volume = df["volume"]
        spot = float(close.iloc[-1])
        rolling_high = float(close.tail(self.LOOKBACK_DAYS).max())
        if rolling_high <= 0:
            return None

        ratio = spot / rolling_high

        # Liquidity filter
        avg_dollar_vol = float((close.tail(30) * volume.tail(30)).mean())
        if avg_dollar_vol < self.MIN_DOLLAR_VOLUME:
            return None

        if ratio < self.NEAR_HIGH_THRESHOLD:
            return None
        if ratio >= self.NEAR_HIGH_CEILING:
            # at or above prior 52w high — avoid blow-off-top buys
            return None

        # Confidence: tighter to the high → higher confidence, scaled in [0.95, 1.0)
        confidence = float(min(0.60 + 4.0 * (ratio - self.NEAR_HIGH_THRESHOLD), 0.95))

        return Signal(
            symbol=symbol,
            side="buy",
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=spot,
            take_profit=round(rolling_high * 1.05, 4),
            stop_loss=round(spot * 0.92, 4),  # 8% stop
            metadata={
                "strategy": self.name,
                "ratio_to_52w_high": round(ratio, 4),
                "52w_high": round(rolling_high, 4),
                "avg_dollar_volume": round(avg_dollar_vol, 0),
                "lookback_days": self.LOOKBACK_DAYS,
                "near_threshold": self.NEAR_HIGH_THRESHOLD,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.LOOKBACK_DAYS + 5:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )

        close = df["close"].astype(float)
        rolling_high = close.rolling(self.LOOKBACK_DAYS, min_periods=120).max().shift(1)
        ratio = (close.shift(1) / rolling_high).replace([np.inf, -np.inf], np.nan)

        entries = ((ratio >= self.NEAR_HIGH_THRESHOLD) & (ratio < self.NEAR_HIGH_CEILING)).fillna(False)
        # Exit when ratio drops below 0.85 (broken anchor)
        exits = (ratio < 0.85).fillna(False)

        return BacktestSignals(entries=entries, exits=exits)