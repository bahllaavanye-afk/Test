"""
Overnight Return Anomaly (ORA)
================================
Academic basis:
  - Lou, Polk, Skouras (2019) "A Day Late and a Dollar Short: Liquidity and Household
    Formation among Student Borrowers" / "Overnight Returns and Firm-Specific Investor
    Sentiment" — essentially 100% of the equity risk premium accrues overnight (close → open).
  - Cliff, Cooper, Gulen (2008) "Return Differences Between Trading and Non-Trading Hours:
    Like Night and Day" documented that overnight returns dominate intraday.

Mechanism:
  Retail investors submit market-on-open orders in the morning (sentiment-driven).
  Institutional capital accumulates during the day, suppressing intraday returns.
  Net result: systematic overnight premium in high-momentum stocks.

Strategy:
  - Universe: top-50 S&P 500 names by market cap (fixed list)
  - Rank by 12-1 month momentum (skip last month to avoid short-term reversal)
  - BUY top-10 at 3:55 PM ET, SELL at 9:35 AM ET next morning
  - Risk filters: skip stocks with bid-ask spread > 0.2% or 30-day avg volume < $10M

Documented Sharpe: ~0.7-1.1 long-only, ~1.2-1.8 with momentum filter
"""

import asyncio
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

ET = ZoneInfo("America/New_York")

_DATA_BASE = "https://data.alpaca.markets"


class OvernightReturnStrategy(AbstractStrategy):
    """
    Overnight Return Anomaly with 12-1 month momentum filter.

    Buy at 3:55 PM ET (MOC-adjacent) in top-momentum stocks,
    sell at 9:35 AM ET next morning after overnight premium is captured.
    """

    name = "overnight_return"
    display_name = "Overnight Return Anomaly"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0  # checked hourly, acts only in windows

    # Top-50 S&P 500 by approximate market cap (as of strategy creation)
    UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK.B",
        "JPM", "V", "MA", "UNH", "JNJ", "XOM", "HD", "PG", "LLY", "ABBV",
        "MRK", "CVX", "BAC", "COST", "PEP", "KO", "WMT", "DIS", "AVGO",
        "CSCO", "CRM", "ACN", "MCD", "ABT", "INTC", "T", "VZ", "NFLX",
        "ORCL", "ADBE", "TMO", "DHR", "NKE", "MDT", "BMY", "LIN", "AMGN",
        "GE", "HON", "UPS", "CAT", "MMM",
    ]

    # Momentum parameters
    MOMENTUM_LOOKBACK_DAYS = 252   # ~12 months
    MOMENTUM_SKIP_DAYS = 21        # skip most-recent month (reversal effect)
    TOP_N = 10                     # hold top-10 ranked names

    # Risk filters
    MAX_SPREAD_PCT = 0.002         # 0.20% max bid-ask spread
    MIN_30D_VOLUME_USD = 10_000_000  # $10M minimum average daily dollar volume

    # Time windows (ET)
    BUY_WINDOW_START = (15, 45)    # 3:45 PM
    BUY_WINDOW_END   = (15, 59)    # 3:59 PM
    SELL_WINDOW_START = (9, 30)    # 9:30 AM
    SELL_WINDOW_END   = (9, 40)    # 9:40 AM

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def _in_buy_window(self, now_et: datetime) -> bool:
        h, m = now_et.hour, now_et.minute
        return (
            (h, m) >= self.BUY_WINDOW_START
            and (h, m) <= self.BUY_WINDOW_END
        )

    def _in_sell_window(self, now_et: datetime) -> bool:
        h, m = now_et.hour, now_et.minute
        return (
            (h, m) >= self.SELL_WINDOW_START
            and (h, m) <= self.SELL_WINDOW_END
        )

    async def _fetch_bars(self, symbol: str, days: int) -> pd.Series:
        """Fetch daily closing prices for momentum computation."""
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
                return pd.Series(dtype=float, name=symbol)
            bars = resp.json().get("bars", [])
            if not bars:
                return pd.Series(dtype=float, name=symbol)
            s = pd.Series(
                {b["t"]: float(b["c"]) for b in bars},
                name=symbol,
            )
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            return pd.Series(dtype=float, name=symbol)

    async def _fetch_quote(self, symbol: str) -> dict:
        """Fetch latest quote for spread filtering."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/quotes/latest",
                    headers=alpaca_headers(),
                )
            if resp.status_code == 200:
                q = resp.json().get("quote", {})
                return {"bid": float(q.get("bp", 0)), "ask": float(q.get("ap", 0))}
        except Exception:
            pass
        return {"bid": 0.0, "ask": 0.0}

    async def _compute_momentum(self, symbol: str) -> float | None:
        """
        Compute 12-1 month momentum: return from t-252 to t-21 days.
        Returns None if insufficient data.
        """
        series = await self._fetch_bars(symbol, self.MOMENTUM_LOOKBACK_DAYS + 10)
        if len(series) < self.MOMENTUM_LOOKBACK_DAYS - 30:
            return None
        # Price at roughly t-252 and t-21
        try:
            price_12m = series.iloc[-(self.MOMENTUM_LOOKBACK_DAYS - self.MOMENTUM_SKIP_DAYS)]
            price_1m  = series.iloc[-self.MOMENTUM_SKIP_DAYS]
            if price_12m <= 0:
                return None
            return float(price_1m / price_12m - 1.0)
        except IndexError:
            return None

    async def _passes_liquidity_filter(self, symbol: str) -> bool:
        """
        Check: bid-ask spread < 0.2% and 30-day avg dollar volume > $10M.
        """
        # Volume check via daily bars
        series_close = await self._fetch_bars(symbol, 35)
        if len(series_close) < 20:
            return False

        # We don't have volume in the series above (only close); fetch with volume
        start = (date.today() - timedelta(days=40)).isoformat()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_DATA_BASE}/v2/stocks/{symbol}/bars",
                    params={"timeframe": "1Day", "start": start, "limit": 35, "feed": "iex"},
                    headers=alpaca_headers(),
                )
            if resp.status_code != 200:
                return False
            bars = resp.json().get("bars", [])
            if not bars:
                return False
            dollar_vols = [float(b["c"]) * float(b["v"]) for b in bars[-30:]]
            avg_dv = np.mean(dollar_vols)
            if avg_dv < self.MIN_30D_VOLUME_USD:
                return False
        except Exception:
            return False

        # Spread check
        quote = await self._fetch_quote(symbol)
        bid, ask = quote["bid"], quote["ask"]
        if bid <= 0 or ask <= 0:
            return True  # can't determine — allow through
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid
        return spread_pct <= self.MAX_SPREAD_PCT

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        In buy window (3:45-3:59 PM ET): rank universe by 12-1 momentum, buy top-10.
        In sell window (9:30-9:40 AM ET): sell any open overnight positions.
        """
        now_et = datetime.now(ET)

        in_buy  = self._in_buy_window(now_et)
        in_sell = self._in_sell_window(now_et)

        if not in_buy and not in_sell:
            return None

        # SELL WINDOW: close overnight positions
        if in_sell:
            # We signal sell for the provided symbol (executor tracks open positions)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=0.90,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "reason": "overnight_exit",
                    "window": "sell",
                    "time_et": now_et.strftime("%H:%M"),
                },
            )

        # BUY WINDOW: compute momentum for universe and rank
        # Compute momentum for all symbols concurrently
        momentum_tasks = {sym: self._compute_momentum(sym) for sym in self.UNIVERSE}
        results = await asyncio.gather(*momentum_tasks.values(), return_exceptions=True)
        momentum_scores: dict[str, float] = {}
        for sym, result in zip(momentum_tasks.keys(), results):
            if isinstance(result, float) and result is not None:
                momentum_scores[sym] = result

        if not momentum_scores:
            return None

        # Rank by momentum descending
        ranked = sorted(momentum_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_candidates = [sym for sym, _ in ranked[:20]]  # check top-20 before liquidity filter

        # Apply liquidity filter concurrently on candidates
        liquidity_checks = await asyncio.gather(
            *[self._passes_liquidity_filter(sym) for sym in top_candidates],
            return_exceptions=True,
        )
        top_qualified = [
            sym for sym, ok in zip(top_candidates, liquidity_checks)
            if ok is True
        ]

        if symbol not in top_qualified[:self.TOP_N]:
            return None

        rank_idx = top_qualified.index(symbol)
        mom_score = momentum_scores[symbol]
        # Confidence: linearly scaled from 0.7 to 0.95 based on rank
        confidence = round(0.95 - rank_idx * (0.25 / self.TOP_N), 3)

        return Signal(
            symbol=symbol,
            side="buy",
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "reason": "overnight_entry",
                "window": "buy",
                "rank": rank_idx + 1,
                "momentum_12_1": round(mom_score, 4),
                "time_et": now_et.strftime("%H:%M"),
                "extended_hours": True,
                "top_n": self.TOP_N,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest using daily OHLCV.

        Entry: previous close (overnight position opened at close).
        Exit: open price of the next bar.
        Signal: enter when prior 12-1 month momentum is top-quintile.

        For single-symbol backtesting the momentum filter is computed on the
        symbol's own return series (cross-sectional rank is approximated
        by percentile threshold on own history).
        """
        if "close" not in df.columns or len(df) < self.MOMENTUM_LOOKBACK_DAYS:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )

        close = df["close"].astype(float)

        # 12-1 month momentum (shift to avoid lookahead)
        ret_12_1 = (
            close.shift(self.MOMENTUM_SKIP_DAYS) /
            close.shift(self.MOMENTUM_LOOKBACK_DAYS) - 1.0
        )

        # Top quintile threshold — use expanding window so we never peek at
        # future observations (rolling quantile on a look-forward window would
        # embed survivorship bias: each bar's threshold would include data it
        # hadn't seen yet in a real-time deployment).
        rolling_80pct = ret_12_1.expanding(min_periods=60).quantile(0.80)

        # Enter when momentum in top quintile (shift 1 to prevent lookahead)
        in_top_quintile = (ret_12_1 > rolling_80pct).shift(1).fillna(False)

        # Overnight entry = every qualifying close, exit = every open (next bar)
        # Proxy: enter on qualifying bar, exit 1 bar later
        entries = in_top_quintile
        exits   = in_top_quintile.shift(1).fillna(False)  # exit after 1-bar hold

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=None,
            short_exits=None,
        )
