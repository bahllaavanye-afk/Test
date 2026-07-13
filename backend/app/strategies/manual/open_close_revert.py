"""
Opening Gap Mean-Reversion (Liquidity Provision)
==================================================
Source: Bogousslavsky (2016) "Infrequent Rebalancing, Return Autocorrelation,
and Seasonality", Journal of Finance.

Bogousslavsky shows that opening prices are systematically noisier than
intraday prices because overnight order imbalances — accumulated by retail
"market-on-open" orders, ETF creation/redemption flows, and forced rebalancers
— land at the open auction without sufficient market-making capital to fully
absorb them. The noise reverts during the day.

Trading implication: an unusually large overnight GAP (down or up) is usually
followed by a partial intraday reversion toward the prior close. A short-term
liquidity provider can monetise this.

Strategy:
  - Universe: top-50 S&P 500 names (high volume, tight spreads).
  - At 9:31-9:35 ET, measure gap_pct = (open - prev_close) / prev_close and
    first-five-minutes volume vs 20-day avg.
  - LONG  entry: gap_pct < -1.0%  AND  first_5m_volume > 1.5 × 20-day-avg-5m
                 (real panic → likely overshoot)
  - SHORT entry: gap_pct > +1.5%  AND  first_5m_volume > 1.5 × 20-day-avg-5m
                 (over-eager retail buying → likely fade)
  - Exit: 15:50-15:55 ET (close auction) — no overnight risk.

Critically: when no fresh news catalyst justifies the gap, the reversion is
statistically large. We approximate "no catalyst" by requiring the gap to be
within 3× recent realised vol — extreme news gaps are typically larger still.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

ET = ZoneInfo("America/New_York")
_DATA_BASE = "https://data.alpaca.markets"


class OpenCloseRevertStrategy(AbstractStrategy):
    name = "open_close_revert"
    display_name = "Opening Gap Reversion (Bogousslavsky 2016)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 60.0  # intraday

    UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "JPM", "V", "MA", "UNH", "JNJ", "XOM", "HD", "PG", "LLY",
        "AVGO", "CRM", "ORCL", "AMD", "COST", "TMO", "AMAT", "MU",
        "BAC", "MRK", "CVX", "ABBV", "MCD", "ABT", "ACN", "ADBE",
        "NKE", "CSCO", "AMGN", "GE", "HON", "UPS", "CAT", "NFLX",
        "PEP", "KO", "WMT", "DIS", "T", "VZ", "INTC", "QCOM", "TXN", "IBM",
    ]

    GAP_DOWN_THRESHOLD = -0.010  # -1.0%
    GAP_UP_THRESHOLD = 0.015     # +1.5%
    VOLUME_MULT = 1.5            # first-5m volume vs 20-day avg
    MAX_GAP_VS_VOL = 3.0         # gap_pct / daily_vol must be < this (else likely news)

    # ET time windows
    BUY_WINDOW_START = (9, 31)
    BUY_WINDOW_END = (9, 35)
    EXIT_WINDOW_START = (15, 50)
    EXIT_WINDOW_END = (15, 55)

    # Simple in‑memory caches to avoid repeated HTTP calls within the same day
    _daily_cache: dict[str, tuple[date, pd.DataFrame]] = {}
    _intraday_cache: dict[str, tuple[date, pd.DataFrame]] = {}

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def _in_window(self, now_et: datetime, start: tuple[int, int], end: tuple[int, int]) -> bool:
        hm = (now_et.hour, now_et.minute)
        return start <= hm <= end

    async def _fetch_daily(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch last `days` daily bars for `symbol`, using a per‑day cache."""
        today = date.today()
        cache_entry = self._daily_cache.get(symbol)
        if cache_entry and cache_entry[0] == today:
            return cache_entry[1]

        start = (today - timedelta(days=days + 10)).isoformat()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "1Day",
                        "start": start,
                        "limit": days + 10,
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
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low": float(b["l"]),
                        "close": float(b["c"]),
                        "volume": float(b.get("v", 0)),
                    }
                    for b in bars
                ]
            )
            df["t"] = pd.to_datetime(df["t"])
            df = df.set_index("t").sort_index()
            # Store in cache
            self._daily_cache[symbol] = (today, df)
            return df
        except Exception:
            return pd.DataFrame()

    async def _fetch_5min_today(self, symbol: str) -> pd.DataFrame:
        """Fetch today's 5‑minute bars for `symbol`, using a per‑day cache."""
        today = date.today()
        cache_entry = self._intraday_cache.get(symbol)
        if cache_entry and cache_entry[0] == today:
            return cache_entry[1]

        start = today.isoformat()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "5Min",
                        "start": start,
                        "limit": 100,
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
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low": float(b["l"]),
                        "close": float(b["c"]),
                        "volume": float(b.get("v", 0)),
                    }
                    for b in bars
                ]
            )
            df["t"] = pd.to_datetime(df["t"])
            df = df.set_index("t").sort_index()
            self._intraday_cache[symbol] = (today, df)
            return df
        except Exception:
            return pd.DataFrame()

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if symbol not in self.UNIVERSE:
            return None

        now_et = datetime.now(ET)
        in_buy = self._in_window(now_et, self.BUY_WINDOW_START, self.BUY_WINDOW_END)
        in_exit = self._in_window(now_et, self.EXIT_WINDOW_START, self.EXIT_WINDOW_END)

        if not in_buy and not in_exit:
            return None

        if in_exit:
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=0.90,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": self.name,
                    "reason": "intraday_exit",
                    "window": "exit",
                    "time_et": now_et.strftime("%H:%M"),
                },
            )

        # BUY window – compute gap and volume metrics
        daily = await self._fetch_daily(symbol, days=30)
        if daily.empty or len(daily) < 5:
            return None
        prev_close = float(daily["close"].iloc[-1])  # last completed session

        intraday = await self._fetch_5min_today(symbol)
        if intraday.empty:
            return None

        today_open = float(intraday["open"].iloc[0])
        first_5m_volume = float(intraday["volume"].iloc[0])

        avg_daily_vol = float(daily["volume"].tail(20).mean())
        baseline_5m = (avg_daily_vol / 78.0) * 3.0
        if baseline_5m <= 0:
            return None

        volume_ratio = first_5m_volume / baseline_5m
        gap_pct = (today_open - prev_close) / prev_close

        # Volatility filter – approximate recent daily vol using close‑to‑close std
        recent_vol = float(daily["close"].pct_change().tail(20).std())
        if recent_vol == 0:
            return None
        if abs(gap_pct) > self.MAX_GAP_VS_VOL * recent_vol:
            return None

        # Determine side based on gap direction and volume multiplier
        if gap_pct <= self.GAP_DOWN_THRESHOLD and volume_ratio >= self.VOLUME_MULT:
            side = "buy"
        elif gap_pct >= self.GAP_UP_THRESHOLD and volume_ratio >= self.VOLUME_MULT:
            side = "sell"
        else:
            return None

        return Signal(
            symbol=symbol,
            side=side,
            confidence=0.85,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": self.name,
                "reason": "gap_reversion",
                "gap_pct": round(gap_pct, 5),
                "volume_ratio": round(volume_ratio, 3),
                "time_et": now_et.strftime("%H:%M"),
            },
        )