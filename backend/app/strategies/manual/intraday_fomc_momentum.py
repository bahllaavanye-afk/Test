"""
Intraday FOMC Momentum
========================
Academic basis:
  - Lucca & Moench (2015) "The Pre-FOMC Announcement Drift," Journal of Finance
    — documents a robust ~50 bps excess return in U.S. equities during the 24-hour
    window preceding scheduled FOMC announcements, present from 1994 onward and
    statistically significant across multiple asset classes.
  - Cieslak, Morse & Vissing-Jorgensen (2019) "Stock Returns over the FOMC Cycle,"
    Journal of Finance — confirms the pre-FOMC drift and shows that stock returns
    are systematically higher in even weeks of the FOMC cycle, consistent with
    information leakage or risk-premium dynamics around scheduled Fed meetings.

Two-component strategy:

1. Pre-FOMC drift (calendar signal):
   - Buy SPY/QQQ on the trading day before a known FOMC date.
   - Exit on the FOMC date itself (hold ~1 day).
   - Documented edge: ~50 bps per event, ~8 events/year.

2. Post-announcement momentum (intraday signal):
   - After the FOMC announcement (~14:00 ET), the initial 5-minute price move in
     SPY predicts the subsequent 60-minute direction (momentum, not reversal).
   - In live mode: check direction of move from 14:00 to 14:05 ET, hold until 15:05 ET.

Backtest (daily bars):
   - Entry: bar on the day BEFORE any FOMC date.
   - Exit: bar ON the FOMC date.
   - Shift(1) applied so signal fires on the close of the "day before" and
     the position is entered at the NEXT open (standard backtest convention).
"""

from datetime import date, timedelta

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

# ---------------------------------------------------------------------------
# Known FOMC meeting dates (announcement day) — 2024 and 2025.
# Source: Federal Reserve press releases / FOMC calendar.
# ---------------------------------------------------------------------------
FOMC_DATES_2024_2025: list[str] = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
]

# Convert to a set of date objects for O(1) lookup
_FOMC_DATE_SET: set[date] = {
    date.fromisoformat(d) for d in FOMC_DATES_2024_2025
}

# Pre-FOMC drift parameters
PRE_FOMC_CONFIDENCE = 0.65
PRE_FOMC_TARGET_RETURN = 0.005   # 50 bps documented drift


class IntradayFOMCMomentumStrategy(AbstractStrategy):
    """
    Calendar-driven strategy exploiting the pre-FOMC announcement drift.

    Buys SPY or QQQ on FOMC-eve, exits on announcement day.
    Documented excess return: ~50 bps per event over 8 scheduled meetings/year.
    """

    name = "intraday_fomc_momentum"
    display_name = "Intraday FOMC Momentum"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0   # hourly check is sufficient (calendar-driven)
    confidence_threshold = 0.60

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        # Allow injecting additional FOMC dates via params for forward extension
        extra_dates: list[str] = (params or {}).get("extra_fomc_dates", [])
        self._fomc_set: set[date] = _FOMC_DATE_SET | {
            date.fromisoformat(d) for d in extra_dates
        }

    def description(self) -> str:
        return (
            f"{self.display_name} — buys SPY/QQQ the day before each FOMC meeting "
            f"(pre-announcement drift, ~50 bps documented edge). "
            f"Exits on announcement day."
        )

    def _is_fomc_eve(self, today: date) -> bool:
        """Return True if tomorrow (next calendar day) is a known FOMC date."""
        return (today + timedelta(days=1)) in self._fomc_set

    def _is_fomc_day(self, today: date) -> bool:
        """Return True if today is a known FOMC announcement date."""
        return today in self._fomc_set

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Check whether today is the day before a known FOMC meeting.
        If yes, emit a bullish signal with confidence=0.65 and target=current_price*1.005.

        Also checks for same-day post-announcement momentum:
        If today is an FOMC day and the 'fomc_5min_move' key is in data metadata,
        use that to emit a directional signal.
        """
        today = date.today()

        # --- Pre-FOMC drift signal ---
        if self._is_fomc_eve(today):
            if data.empty or "close" not in data.columns:
                return None
            current_price = float(data["close"].iloc[-1])
            target = round(current_price * (1.0 + PRE_FOMC_TARGET_RETURN), 4)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=PRE_FOMC_CONFIDENCE,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=target,
                metadata={
                    "signal_type": "pre_fomc_drift",
                    "fomc_date": (today + timedelta(days=1)).isoformat(),
                    "current_price": current_price,
                    "documented_edge_bps": 50,
                },
            )

        # --- Post-announcement 5-minute momentum signal ---
        if self._is_fomc_day(today) and not data.empty and "close" in data.columns:
            # Requires caller to supply recent intraday bars around 14:00–14:05 ET.
            # We use the last two bars to infer the 5-minute direction.
            if len(data) >= 2:
                move_5min = float(data["close"].iloc[-1]) / float(data["close"].iloc[-2]) - 1.0
                current_price = float(data["close"].iloc[-1])
                if abs(move_5min) >= 0.001:   # at least 10 bps to act on
                    side = "buy" if move_5min > 0 else "sell"
                    confidence = min(0.80, 0.60 + abs(move_5min) * 20)
                    target = round(
                        current_price * (1 + abs(move_5min) * 3) if side == "buy"
                        else current_price * (1 - abs(move_5min) * 3),
                        4,
                    )
                    return Signal(
                        symbol=symbol,
                        side=side,
                        confidence=round(confidence, 4),
                        strategy_name=self.name,
                        strategy_type=self.strategy_type,
                        risk_bucket=self.risk_bucket,
                        target_price=target,
                        metadata={
                            "signal_type": "post_fomc_5min_momentum",
                            "fomc_date": today.isoformat(),
                            "5min_move_pct": round(move_5min * 100, 4),
                            "current_price": current_price,
                        },
                    )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest on daily OHLCV data.

        For each bar:
          - Mark whether the NEXT calendar day is an FOMC date (this is FOMC-eve).
          - Entry: bar is FOMC-eve (buy at next open).
          - Exit: bar is on the FOMC date itself (sell at next open).
          - shift(1) ensures no lookahead — the signal fires on the close of
            FOMC-eve and the position opens on the following bar.

        Produces sparse signals (~8 per year) each carrying the documented
        pre-announcement edge.
        """
        required = {"close"}
        if not required.issubset(df.columns) or len(df) < 3:
            empty = pd.Series(False, index=df.index, dtype=bool)
            return BacktestSignals(entries=empty, exits=empty)

        # Normalise index to date objects for comparison
        if hasattr(df.index, "date"):
            index_dates = pd.Series(df.index.date, index=df.index)
        else:
            # Try parsing if it's a string/object index
            index_dates = pd.Series(
                pd.to_datetime(df.index).date, index=df.index
            )

        # FOMC-eve: the day immediately before a known FOMC date
        fomc_eve = index_dates.apply(
            lambda d: (d + timedelta(days=1)) in self._fomc_set
        )

        # FOMC day itself
        fomc_day = index_dates.apply(lambda d: d in self._fomc_set)

        # shift(1): signal known at end-of-bar, acted on next bar
        entries = fomc_eve.shift(1).fillna(False).astype(bool)
        exits = fomc_day.shift(1).fillna(False).astype(bool)

        return BacktestSignals(entries=entries, exits=exits)
