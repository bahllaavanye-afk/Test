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
    MAX_GAP_VS_VOL = 2.5         # MUTATION: tightened from 3.0 to reduce false positives from news‑driven gaps
    # ET time windows
    BUY_WINDOW_START = (9, 31)
    BUY_WINDOW_END = (9, 35)
    EXIT_WINDOW_START = (15, 50)
    EXIT_WINDOW_END = (15, 55)

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def _in_window(self, now_et: datetime, start: tuple[int, int], end: tuple[int, int]) -> bool:
        hm = (now_et.hour, now_et.minute)
        return start <= hm <= end

    async def _fetch_daily(self, symbol: str, days: int = 30) -> pd.DataFrame:
        start = (date.today() - timedelta(days=days + 10)).isoformat()
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
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"])
            df.set_index("t", inplace=True)
            return df
        except Exception:
            return pd.DataFrame()

    async def _fetch_intraday(self, symbol: str, days: int = 1) -> pd.DataFrame:
        start = (date.today() - timedelta(days=days)).isoformat()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": "1Min",
                        "start": start,
                        "limit": 390 * days,
                        "feed": "iex",
                    },
                    headers=alpaca_headers(),
                )
            if resp.status_code != 200:
                return pd.DataFrame()
            bars = resp.json().get("bars", [])
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"])
            df.set_index("t", inplace=True)
            return df
        except Exception:
            return pd.DataFrame()

    async def generate_signals(self, now: datetime) -> BacktestSignals:
        signals = []
        now_et = now.astimezone(ET)

        # Only generate signals within the entry window
        if not self._in_window(now_et, self.BUY_WINDOW_START, self.BUY_WINDOW_END):
            return BacktestSignals(signals)

        for symbol in self.UNIVERSE:
            daily_df = await self._fetch_daily(symbol, days=30)
            if daily_df.empty or len(daily_df) < 2:
                continue

            # Compute recent daily volatility (standard deviation of returns)
            daily_df["ret"] = daily_df["c"].pct_change()
            recent_vol = daily_df["ret"].std()

            # Get previous close and today's open
            prev_close = daily_df["c"].iloc[-2]
            today_open = daily_df["o"].iloc[-1]

            gap_pct = (today_open - prev_close) / prev_close

            # Filter out news‑driven gaps
            if abs(gap_pct) / recent_vol > self.MAX_GAP_VS_VOL:
                continue

            # Fetch intraday 5‑minute volume
            intraday_df = await self._fetch_intraday(symbol, days=20)
            if intraday_df.empty:
                continue

            # Compute 20‑day average 5‑minute volume
            intraday_df["date"] = intraday_df.index.date
            vol_5m = intraday_df.groupby("date")["v"].sum() / (6.5 * 60)  # approximate per‑minute volume
            avg_5m_vol = vol_5m.tail(20).mean()

            # Volume in the first 5 minutes of today
            first_5m = intraday_df.between_time("09:31", "09:35")
            if first_5m.empty:
                continue
            today_5m_vol = first_5m["v"].sum()

            if today_5m_vol < self.VOLUME_MULT * avg_5m_vol:
                continue

            if gap_pct <= self.GAP_DOWN_THRESHOLD:
                signals.append(Signal(symbol=symbol, side="long", qty=100))
            elif gap_pct >= self.GAP_UP_THRESHOLD:
                signals.append(Signal(symbol=symbol, side="short", qty=100))

        return BacktestSignals(signals)

    async def generate_exit_signals(self, now: datetime) -> BacktestSignals:
        now_et = now.astimezone(ET)

        # Only exit during the exit window
        if not self._in_window(now_et, self.EXIT_WINDOW_START, self.EXIT_WINDOW_END):
            return BacktestSignals([])

        exit_signals = [
            Signal(symbol=symbol, side="flat", qty=0) for symbol in self.UNIVERSE
        ]
        return BacktestSignals(exit_signals)
