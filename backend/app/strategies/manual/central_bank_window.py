"""
Central Bank Window Strategy
==============================
Academic basis:
  - Lucca & Moench (2015, JF): "The Pre-FOMC Announcement Drift" — US equities
    earn abnormal returns in the 24h window BEFORE FOMC announcements. SPY
    gains ~50bps on average in the day before (4pm ET prior day to 2pm ET FOMC day).
  - Extension: same drift observed for ECB, BOJ announcements on global ETFs.
  - The drift is attributed to dealer hedging demand and informed pre-positioning.

Implementation:
  - Hard-coded Fed meeting calendar for 2025-2026 (8 meetings/year, published by Fed).
  - Buy SPY at close of day T-1 (day before FOMC). Sell at 2pm ET on day T (FOMC day).
  - Use Alpaca paper account. Entry/exit via limit-first execution.

Approximate Sharpe: 1.5-2.0 on SPY (Lucca & Moench backtest 1994-2011)
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

# FOMC scheduled meeting dates (all 8 meetings per year) — date of the announcement (day T)
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

# Projected 2027 dates (typical Fed schedule: 8 meetings at 6-7 week intervals)
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
    for d in ALL_FOMC_DATES:
        if d >= today:
            return d
    return None


class CentralBankWindowStrategy(AbstractStrategy):
    """
    Pre-FOMC announcement drift: buy SPY at close day T-1, sell at open day T.
    Academic: Lucca & Moench (2015, JF) — earns ~50bps per meeting, Sharpe ~2.0.
    Risk bucket: directional, market_type: equity
    """

    name = "central_bank_window"
    display_name = "Central Bank Window (Pre-FOMC Drift)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0  # hourly checks

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        today = date.today()
        trade_sym = symbol if symbol in ("SPY", "QQQ") else "SPY"

        next_fomc = _next_fomc(today)
        if next_fomc is None:
            import logging
            logging.getLogger(__name__).warning(
                "central_bank_window: FOMC calendar exhausted — no dates after %s. "
                "Update ALL_FOMC_DATES in central_bank_window.py.", today
            )
            return None

        entry_day = next_fomc - timedelta(days=1)
        # Skip weekends
        while entry_day.weekday() > 4:
            entry_day -= timedelta(days=1)

        if today == entry_day:
            return Signal(
                symbol=trade_sym,
                side="buy",
                confidence=0.78,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "event": "pre_fomc_drift_entry",
                    "fomc_date": str(next_fomc),
                    "entry_day": str(entry_day),
                    "academic_ref": "Lucca & Moench (2015, JF) Pre-FOMC Announcement Drift",
                    "expected_return_bps": 50,
                },
            )

        if today == next_fomc:
            return Signal(
                symbol=trade_sym,
                side="sell",
                confidence=0.85,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "event": "pre_fomc_drift_exit",
                    "fomc_date": str(next_fomc),
                    "academic_ref": "Lucca & Moench (2015, JF)",
                },
            )

        return None

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
        exits   = pd.Series([d in fomc_set  for d in dates], index=df.index)

        return BacktestSignals(entries=entries, exits=exits)
