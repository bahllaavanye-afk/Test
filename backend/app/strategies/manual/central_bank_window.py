"""
Central Bank Window Strategy
==============================
Academic basis:
  - Lucca & Moench (2015, JF): "The Pre-FOMC Announcement Drift" — US equities
    earn abnormal returns in the 24h window BEFORE FOMC announcements. SPY
    gains ~50bps on average in the day before (4pm ET prior day to 2pm ET FOMC day).
  - Extension: same drift observed for ECB, BOJ announcements on global ETFs.
  - The drift is attributed to dealer hedging demand and informed pre‑positioning.

Implementation:
  - Hard‑coded Fed meeting calendar for 2025‑2027 (8 meetings/year, published by Fed).
  - Buy SPY at close of day T‑1 (day before FOMC) when market conditions are
    supportive; sell at close of day T (FOMC day) with a simple confirmation filter.
  - Use Alpaca paper account. Entry/exit via limit‑first execution.

Approximate Sharpe: 1.5‑2.0 on SPY (Lucca & Moench backtest 1994‑2011)
"""

from __future__ import annotations

from datetime import date, timedelta
import logging

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

# ----------------------------------------------------------------------
# Fed meeting dates (announcement day, T)
# ----------------------------------------------------------------------
_FOMC_DATES_2025 = [
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
]

_FOMC_DATES_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 5, 6),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]

_FOMC_DATES_2027 = [
    date(2027, 1, 27),
    date(2027, 3, 17),
    date(2027, 5, 5),
    date(2027, 6, 16),
    date(2027, 7, 28),
    date(2027, 9, 15),
    date(2027, 10, 27),
    date(2027, 12, 8),
]

ALL_FOMC_DATES = sorted(_FOMC_DATES_2025 + _FOMC_DATES_2026 + _FOMC_DATES_2027)


def _next_fomc(today: date) -> date | None:
    """Return the next FOMC announcement date on or after *today*."""
    for d in ALL_FOMC_DATES:
        if d >= today:
            return d
    return None


class CentralBankWindowStrategy(AbstractStrategy):
    """
    Pre‑FOMC announcement drift: buy SPY at close day T‑1, sell at close day T.
    Academic: Lucca & Moench (2015, JF) — earns ~50bps per meeting, Sharpe ~2.0.
    Risk bucket: directional, market_type: equity
    """

    name = "central_bank_window"
    display_name = "Central Bank Window (Pre‑FOMC Drift)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0  # hourly checks

    # ------------------------------------------------------------------
    # Helper methods for signal quality
    # ------------------------------------------------------------------
    @staticmethod
    def _price_above_sma(close_series: pd.Series, window: int = 20) -> bool:
        """Return True if the latest close is above the *window*‑day SMA."""
        if len(close_series) < window:
            return False
        sma = close_series.rolling(window).mean().iloc[-1]
        return close_series.iloc[-1] > sma

    @staticmethod
    def _low_volatility(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> bool:
        """
        Compute a simple ATR‑based volatility measure.
        Return True if ATR / price < 0.5 % (i.e. relatively calm market).
        """
        if len(high) < window:
            return False
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window).mean().iloc[-1]
        price = close.iloc[-1]
        return (atr / price) < 0.005  # 0.5 %

    @staticmethod
    def _positive_day_over_day(close: pd.Series) -> bool:
        """Return True if today's close is higher than yesterday's."""
        if len(close) < 2:
            return False
        return close.iloc[-1] > close.iloc[-2]

    # ------------------------------------------------------------------
    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a buy signal on the trading day preceding a FOMC announcement
        when market conditions are favorable, and a sell signal on the announcement
        day itself. The method uses simple price‑based confirmation filters to
        improve signal quality.
        """
        today = date.today()
        trade_sym = symbol if symbol in ("SPY", "QQQ") else "SPY"

        next_fomc = _next_fomc(today)
        if next_fomc is None:
            logging.getLogger(__name__).warning(
                "central_bank_window: FOMC calendar exhausted — no dates after %s. "
                "Update ALL_FOMC_DATES in central_bank_window.py.", today
            )
            return None

        # ------------------------------------------------------------------
        # Determine entry day (T‑1) and adjust for weekends
        # ------------------------------------------------------------------
        entry_day = next_fomc - timedelta(days=1)
        while entry_day.weekday() > 4:  # 5=Saturday, 6=Sunday
            entry_day -= timedelta(days=1)

        # ------------------------------------------------------------------
        # ENTRY LOGIC – apply confirmation filters
        # ------------------------------------------------------------------
        if today == entry_day:
            # Require sufficient price history for the filters
            required_len = max(20, 10) + 1
            if len(data) < required_len:
                return None

            # Ensure required columns exist
            needed_cols = {"close", "high", "low"}
            if not needed_cols.issubset(data.columns):
                return None

            close = data["close"]
            high = data["high"]
            low = data["low"]

            price_ok = self._price_above_sma(close, window=20)
            vol_ok = self._low_volatility(high, low, close, window=10)

            if not (price_ok and vol_ok):
                # Conditions not met – skip entry to avoid low‑quality signal
                return None

            return Signal(
                symbol=trade_sym,
                side="buy",
                confidence=0.88,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "event": "pre_fomc_drift_entry",
                    "fomc_date": str(next_fomc),
                    "entry_day": str(entry_day),
                    "academic_ref": "Lucca & Moench (2015, JF) Pre‑FOMC Announcement Drift",
                    "expected_return_bps": 50,
                    "filters": {"price_above_sma20": price_ok, "low_volatility": vol_ok},
                },
            )

        # ------------------------------------------------------------------
        # EXIT LOGIC – add a simple confirmation filter
        # ------------------------------------------------------------------
        if today == next_fomc:
            # If we have price data, confirm that today closed higher than yesterday;
            # otherwise fall back to default confidence.
            confidence = 0.85
            if len(data) >= 2 and "close" in data.columns:
                if not self._positive_day_over_day(data["close"]):
                    confidence = 0.60  # weaker conviction if drift is negative

            return Signal(
                symbol=trade_sym,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "event": "pre_fomc_drift_exit",
                    "fomc_date": str(next_fomc),
                    "academic_ref": "Lucca & Moench (2015, JF)",
                    "confidence_adjusted": confidence,
                },
            )

        return None

    # ------------------------------------------------------------------
    # Backtesting helpers – unchanged logic
    # ------------------------------------------------------------------
    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "date" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
        else:
            dates = pd.to_datetime(df["date"]).dt.date

        fomc_set = set(ALL_FOMC_DATES)
        entry_set = set()
        for d in fomc_set:
            pre = d - timedelta(days=1)
            while pre.weekday() > 4:
                pre -= timedelta(days=1)
            entry_set.add(pre)

        entries = pd.Series([d in entry_set for d in dates], index=df.index)
        exits = pd.Series([d in fomc_set for d in dates], index=df.index)

        return BacktestSignals(entries=entries, exits=exits)