"""
Triple-Barrier Labeled Momentum
=================================
Source: Marcos López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 3.

Vanilla momentum uses a single N-month lookback return as both the signal AND the
implicit label. This is statistically weak: it ignores path-dependence, doesn't
distinguish luck from skill, and is contaminated by mean-reverting noise.

López de Prado's TRIPLE-BARRIER LABELING fixes this by setting three barriers
around every candidate entry:
  - PT (profit-take):   +2 × ATR
  - SL (stop-loss):     -1 × ATR
  - T  (vertical time): 20 bars

Whichever barrier is touched first determines the realised label:
  +1  if PT hit first  (clean trend that paid off)
  -1  if SL hit first  (failed momentum)
   0  if T  hit first  (drifted nowhere)

Combined with two trending-regime filters (ADX > 25 and ATR/price > 1.5%), this
gives a momentum signal that fires only when (a) the regime is genuinely trending
and (b) recent historical labels with the same setup were predominantly +1.

Documented behaviour: cleaner equity curve and dramatically lower turnover than
classical N-month momentum, with higher hit-rate at similar gross return.
"""

import asyncio
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.brokers.alpaca_headers import alpaca_headers
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_DATA_BASE = "https://data.alpaca.markets"


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(n, min_periods=n).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = _true_range(high, low, close)
    atr = tr.rolling(n, min_periods=n).mean().replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.rolling(n, min_periods=n).mean() / atr
    minus_di = 100.0 * minus_dm.rolling(n, min_periods=n).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.rolling(n, min_periods=n).mean()


def _triple_barrier_labels(
    close: np.ndarray,
    atr: np.ndarray,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    horizon: int = 20,
) -> np.ndarray:
    """
    Vectorisable-but-correct (looped) triple-barrier labelling for a single asset.
    Returns array of {+1, -1, 0} aligned with the entry index.
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        upper = close[i] + pt_mult * atr[i]
        lower = close[i] - sl_mult * atr[i]
        end = min(i + horizon + 1, n)
        label = 0
        for j in range(i + 1, end):
            if close[j] >= upper:
                label = 1
                break
            if close[j] <= lower:
                label = -1
                break
        labels[i] = label
    return labels


class TripleBarrierMomentumStrategy(AbstractStrategy):
    name = "triple_barrier_momentum"
    display_name = "Triple-Barrier Momentum (López de Prado)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "JPM", "V", "MA", "UNH", "JNJ", "XOM", "HD", "LLY",
        "AVGO", "CRM", "ORCL", "AMD", "COST", "TMO", "AMAT", "MU",
    ]

    ATR_PERIOD = 14
    ADX_PERIOD = 14
    MOMENTUM_LOOKBACK = 20
    HORIZON = 20
    PT_MULT = 2.0
    SL_MULT = 1.0

    ADX_THRESHOLD = 25.0
    ATR_PCT_THRESHOLD = 0.015
    MOMENTUM_RANK_THRESHOLD = 0.70
    POSITIVE_LABEL_FRACTION = 0.50  # of finished-labels in lookback window, fraction = +1

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    async def _fetch_ohlc(self, symbol: str, days: int = 90) -> pd.DataFrame:
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
            df = df.set_index("t").sort_index()
            return df
        except Exception:
            return pd.DataFrame()

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if symbol not in self.UNIVERSE:
            return None

        df = await self._fetch_ohlc(symbol, days=90)
        if df.empty or len(df) < max(self.MOMENTUM_LOOKBACK, self.ADX_PERIOD) + self.HORIZON + 5:
            return None

        high, low, close = df["high"], df["low"], df["close"]
        atr_series = _atr(high, low, close, self.ATR_PERIOD)
        adx_series = _adx(high, low, close, self.ADX_PERIOD)

        spot = float(close.iloc[-1])
        atr_now = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
        adx_now = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0
        if spot <= 0 or atr_now <= 0:
            return None

        atr_pct = atr_now / spot

        # Momentum percentile rank on last 60 bars (current vs rolling distribution)
        momentum = close.pct_change(self.MOMENTUM_LOOKBACK)
        mom_now = float(momentum.iloc[-1]) if pd.notna(momentum.iloc[-1]) else 0.0
        recent_mom = momentum.dropna().tail(60)
        if len(recent_mom) < 10:
            return None
        mom_rank = float((recent_mom < mom_now).mean())

        # Triple-barrier labels on history — only count labels whose horizon has fully closed.
        labels = _triple_barrier_labels(
            close.values.astype(float),
            atr_series.values.astype(float),
            pt_mult=self.PT_MULT,
            sl_mult=self.SL_MULT,
            horizon=self.HORIZON,
        )
        finished = labels[: -self.HORIZON]  # only fully-resolved labels
        finished = finished[-60:]            # use last 60 resolved labels
        if len(finished) < 10:
            return None
        positive_fraction = float(np.mean(finished == 1))

        # Regime + label-history gating
        if adx_now < self.ADX_THRESHOLD:
            return None
        if atr_pct < self.ATR_PCT_THRESHOLD:
            return None
        if mom_rank < self.MOMENTUM_RANK_THRESHOLD:
            return None
        if positive_fraction < self.POSITIVE_LABEL_FRACTION:
            return None

        confidence = float(min(
            0.55
            + 0.20 * ((adx_now - self.ADX_THRESHOLD) / 25.0)
            + 0.15 * (mom_rank - self.MOMENTUM_RANK_THRESHOLD) / (1.0 - self.MOMENTUM_RANK_THRESHOLD + 1e-9)
            + 0.10 * positive_fraction,
            0.99,
        ))

        return Signal(
            symbol=symbol,
            side="buy",
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=spot,
            stop_loss=round(spot - self.SL_MULT * atr_now, 4),
            take_profit=round(spot + self.PT_MULT * atr_now, 4),
            metadata={
                "strategy": self.name,
                "adx": round(adx_now, 2),
                "atr": round(atr_now, 4),
                "atr_pct": round(atr_pct, 4),
                "momentum_20d": round(mom_now, 4),
                "momentum_rank": round(mom_rank, 3),
                "positive_label_fraction": round(positive_fraction, 3),
                "horizon_bars": self.HORIZON,
                "pt_mult": self.PT_MULT,
                "sl_mult": self.SL_MULT,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if not {"high", "low", "close"}.issubset(df.columns) or len(df) < 60:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        atr_series = _atr(high, low, close, self.ATR_PERIOD)
        adx_series = _adx(high, low, close, self.ADX_PERIOD)
        momentum = close.pct_change(self.MOMENTUM_LOOKBACK)
        mom_rank = momentum.rolling(60, min_periods=20).apply(
            lambda w: (w < w.iloc[-1]).mean(),
            raw=False,
        )
        atr_pct = atr_series / close.replace(0.0, np.nan)

        adx_ok = (adx_series.shift(1) > self.ADX_THRESHOLD).fillna(False)
        atr_ok = (atr_pct.shift(1) > self.ATR_PCT_THRESHOLD).fillna(False)
        mom_ok = (mom_rank.shift(1) > self.MOMENTUM_RANK_THRESHOLD).fillna(False)

        entries = adx_ok & atr_ok & mom_ok

        # Exit on triple-barrier: simulate via PT/SL/timer on close path
        close_arr = close.values
        atr_arr = atr_series.values
        n = len(close)
        exits = np.zeros(n, dtype=bool)
        entry_indices = np.where(entries.values)[0]
        for i in entry_indices:
            if not np.isfinite(atr_arr[i]) or atr_arr[i] <= 0:
                continue
            upper = close_arr[i] + self.PT_MULT * atr_arr[i]
            lower = close_arr[i] - self.SL_MULT * atr_arr[i]
            end = min(i + self.HORIZON + 1, n)
            exit_at = end - 1
            for j in range(i + 1, end):
                if close_arr[j] >= upper or close_arr[j] <= lower:
                    exit_at = j
                    break
            if exit_at < n:
                exits[exit_at] = True

        return BacktestSignals(
            entries=entries,
            exits=pd.Series(exits, index=df.index),
        )
