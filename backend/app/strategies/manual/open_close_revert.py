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
    first-five-minutes volume vs 20‑day avg.
  - LONG  entry: gap_pct < -1.0%  AND  first_5m_volume > 1.5 × 20‑day‑avg‑5m
                 (real panic → likely overshoot)
  - SHORT entry: gap_pct > +1.5%  AND  first_5m_volume > 1.5 × 20‑day‑avg‑5m
                 (over‑eager retail buying → likely fade)
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
    """
    Opening‑gap reversion strategy.

    The implementation is defensive: it gracefully handles ``None`` inputs,
    empty data frames, missing columns, and off‑by‑one indexing edge cases.
    """

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
    VOLUME_MULT = 1.5            # first‑5m volume vs 20‑day avg
    MAX_GAP_VS_VOL = 3.0         # gap_pct / daily_vol_std must be < this

    # ET time windows
    BUY_WINDOW_START = (9, 31)
    BUY_WINDOW_END = (9, 35)
    EXIT_WINDOW_START = (15, 50)
    EXIT_WINDOW_END = (15, 55)

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _in_window(self, now_et: datetime, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """Return ``True`` if ``now_et`` falls within the inclusive ``[start, end]`` window."""
        hm = (now_et.hour, now_et.minute)
        return start <= hm <= end

    async def _fetch_daily(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch the last ``days`` + 10 calendar days of daily bars for *symbol*."""
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
            return df.set_index("t").sort_index()
        except Exception:
            return pd.DataFrame()

    async def _fetch_5min_today(self, symbol: str) -> pd.DataFrame:
        """Fetch today's 5‑minute bars for *symbol* (only the early bars are needed)."""
        start = date.today().isoformat()
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
            return df.set_index("t").sort_index()
        except Exception:
            return pd.DataFrame()

    # --------------------------------------------------------------------- #
    # Core analysis
    # --------------------------------------------------------------------- #
    async def analyze(self, data: pd.DataFrame | None, symbol: str) -> Signal | None:
        """
        Produce a trading signal for *symbol*.

        The method is tolerant to ``None``/empty inputs and avoids index errors.
        """
        # Guard against unexpected ``None`` data – the strategy does not depend on it.
        if data is None:
            data = pd.DataFrame()

        if symbol not in self.UNIVERSE:
            return None

        now_et = datetime.now(ET)
        in_buy = self._in_window(now_et, self.BUY_WINDOW_START, self.BUY_WINDOW_END)
        in_exit = self._in_window(now_et, self.EXIT_WINDOW_START, self.EXIT_WINDOW_END)

        # Outside of both windows – nothing to do.
        if not in_buy and not in_exit:
            return None

        # ----------------------------------------------------------------- #
        # EXIT logic – close any open position at the end of day.
        # ----------------------------------------------------------------- #
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

        # ----------------------------------------------------------------- #
        # BUY window – compute entry criteria.
        # ----------------------------------------------------------------- #
        daily = await _fetch_daily(self, symbol, days=30)
        if daily.empty or len(daily) < 5:
            return None

        # Determine the most recent *completed* daily close.
        # If the latest row belongs to today (which may be incomplete), step back one.
        last_idx = daily.index[-1]
        if last_idx.date() == date.today():
            if len(daily) < 2:
                return None
            prev_close = float(daily["close"].iloc[-2])
        else:
            prev_close = float(daily["close"].iloc[-1])

        intraday = await _fetch_5min_today(self, symbol)
        if intraday.empty:
            return None

        # Ensure the first 5‑minute bar exists.
        if intraday.shape[0] == 0:
            return None
        today_open = float(intraday["open"].iloc[0])
        first_5m_volume = float(intraday["volume"].iloc[0])

        # ----------------------------------------------------------------- #
        # Volume baseline – 20‑day average of daily volume, scaled to a 5‑min bar.
        # ----------------------------------------------------------------- #
        avg_daily_vol = float(daily["volume"].tail(20).mean())
        if avg_daily_vol <= 0:
            return None
        baseline_5m = (avg_daily_vol / 78.0) * 3.0  # 78 five‑minute bars per session

        # ----------------------------------------------------------------- #
        # Gap calculation and sanity checks.
        # ----------------------------------------------------------------- #
        if prev_close == 0:
            return None
        gap_pct = (today_open - prev_close) / prev_close

        # Recent realised volatility – standard deviation of daily returns over the last 20 days.
        recent_returns = daily["close"].pct_change().tail(20)
        daily_vol_std = float(recent_returns.std())
        if daily_vol_std <= 0:
            # Avoid division‑by‑zero; treat as non‑trading condition.
            return None
        gap_vs_vol = abs(gap_pct) / daily_vol_std

        # ----------------------------------------------------------------- #
        # Entry criteria.
        # ----------------------------------------------------------------- #
        volume_ok = first_5m_volume > self.VOLUME_MULT * baseline_5m
        gap_ok = gap_vs_vol < self.MAX_GAP_VS_VOL

        # LONG entry
        if gap_pct <= self.GAP_DOWN_THRESHOLD and volume_ok and gap_ok:
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=0.95,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": self.name,
                    "reason": "gap_down_long",
                    "gap_pct": gap_pct,
                    "first_5m_volume": first_5m_volume,
                    "baseline_5m": baseline_5m,
                    "window": "buy",
                    "time_et": now_et.strftime("%H:%M"),
                },
            )

        # SHORT entry
        if gap_pct >= self.GAP_UP_THRESHOLD and volume_ok and gap_ok:
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=0.95,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "strategy": self.name,
                    "reason": "gap_up_short",
                    "gap_pct": gap_pct,
                    "first_5m_volume": first_5m_volume,
                    "baseline_5m": baseline_5m,
                    "window": "buy",
                    "time_et": now_et.strftime("%H:%M"),
                },
            )

        # No entry signal.
        return None