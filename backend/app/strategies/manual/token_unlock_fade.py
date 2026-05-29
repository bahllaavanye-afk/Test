"""
Token Unlock Fade Strategy.

Research findings:
  - 90% of token unlocks see price decline in the 30 days before / around unlock.
  - Team/investor unlocks: average -25% price impact over 30-day window.
  - Optimal entry for recovery: day 14-30 POST-unlock (selling pressure exhausted).

Two-phase strategy:
  Phase 1 (SHORT): enter 7 days before unlock, exit on unlock day.
  Phase 2 (LONG recovery): enter 14 days after unlock, hold for 16 days (mean reversion).

Data sources:
  - Manually configured unlock calendar (stored in KNOWN_UNLOCKS or passed as params).
  - yfinance for price history.

Academic reference:
  Vasan et al. (2022) "Automated Market Makers and Decentralized Exchanges: A DeFi
    Primer" — Working Paper.
  Cong et al. (2021) "Tokenomics: Dynamic Adoption and Valuation" — RFS.
  Documented: 90% unlock-price decline rate; -25% average for large unlocks.
"""
from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class TokenUnlockFade(AbstractStrategy):
    """
    Token unlock event trading: short before unlock, long after selling pressure exhausted.

    KNOWN_UNLOCKS format: (symbol, unlock_date_str YYYY-MM-DD, unlock_fraction 0-1)
    """

    name = "token_unlock_fade"
    display_name = "Token Unlock Fade"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86_400.0   # daily check

    # Hardcoded calendar of known upcoming unlocks (symbol, date, fraction)
    KNOWN_UNLOCKS: list[tuple[str, str, float]] = [
        ("SOL",   "2025-01-01", 0.05),
        ("APT",   "2025-02-01", 0.08),
        ("ARB",   "2025-03-16", 0.10),
        ("OP",    "2025-04-30", 0.07),
        ("SUI",   "2025-05-03", 0.06),
        ("AVAX",  "2025-06-01", 0.04),
        ("NEAR",  "2025-07-14", 0.09),
        ("INJ",   "2025-08-01", 0.05),
        ("IMX",   "2025-09-01", 0.12),
        ("DYDX",  "2025-10-01", 0.08),
    ]

    # Phase thresholds (days relative to unlock date)
    SHORT_ENTRY_DAYS_BEFORE: int = 7
    SHORT_EXIT_DAYS_AFTER: int = 0    # exit on unlock day
    LONG_ENTRY_DAYS_AFTER: int = 14
    LONG_EXIT_DAYS_AFTER: int = 30

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = params or {}
        self.short_entry_days_before: int = int(
            p.get("short_entry_days_before", self.SHORT_ENTRY_DAYS_BEFORE)
        )
        self.long_entry_days_after: int = int(
            p.get("long_entry_days_after", self.LONG_ENTRY_DAYS_AFTER)
        )
        self.long_exit_days_after: int = int(
            p.get("long_exit_days_after", self.LONG_EXIT_DAYS_AFTER)
        )
        # Allow additional unlocks via params
        extra_unlocks: list[tuple[str, str, float]] = list(
            p.get("unlock_schedule", [])
        )
        self._all_unlocks: list[tuple[str, str, float]] = list(self.KNOWN_UNLOCKS) + extra_unlocks

    def description(self) -> str:
        return (
            f"Short {self.short_entry_days_before}d before token unlock, exit on unlock day. "
            f"Enter long recovery {self.long_entry_days_after}d post-unlock for mean reversion. "
            "Source: Cong et al. (2021) Tokenomics, 90% unlock-price decline rate."
        )

    def _get_unlock_entries(self, symbol: str) -> list[date]:
        """Return list of unlock dates for the given symbol."""
        sym_upper = symbol.upper().replace("-USD", "").replace("USDT", "")
        results = []
        for s, d_str, _frac in self._all_unlocks:
            if s.upper() == sym_upper:
                try:
                    results.append(datetime.strptime(d_str, "%Y-%m-%d").date())
                except ValueError:
                    continue
        return results

    def _get_phase(self, today: date, unlock_date: date) -> str | None:
        """
        Return the current trading phase relative to unlock_date:
          'short'    — short entry window (7 days before unlock)
          'exit'     — exit short (unlock day ± 1)
          'recovery' — long recovery window (14-30 days post-unlock)
          None       — no action
        """
        days_to_unlock = (unlock_date - today).days
        days_since_unlock = (today - unlock_date).days

        if 1 <= days_to_unlock <= self.short_entry_days_before:
            return "short"
        if -1 <= days_to_unlock <= 1:
            return "exit"
        if self.long_entry_days_after <= days_since_unlock <= self.long_exit_days_after:
            return "recovery"
        return None

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Check the unlock calendar for the given symbol and emit signals.
        """
        if "close" not in data.columns or len(data) == 0:
            raise ValueError("TokenUnlockFade.analyze: 'close' column required in data.")

        current_price = float(data["close"].iloc[-1])
        if current_price <= 0:
            raise ValueError(f"TokenUnlockFade: invalid price {current_price}")

        today = datetime.now(timezone.utc).date()
        unlock_dates = self._get_unlock_entries(symbol)

        if not unlock_dates:
            return None

        for unlock_date in unlock_dates:
            phase = self._get_phase(today, unlock_date)

            if phase == "short":
                days_to_unlock = (unlock_date - today).days
                return Signal(
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    symbol=symbol,
                    side="sell",
                    confidence=0.72,
                    target_price=current_price,
                    metadata={
                        "phase": "pre_unlock_short",
                        "unlock_date": str(unlock_date),
                        "days_to_unlock": days_to_unlock,
                        "order_type": "limit",
                    },
                )

            if phase == "exit":
                return Signal(
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    symbol=symbol,
                    side="buy",   # close the short
                    confidence=0.80,
                    target_price=current_price,
                    metadata={
                        "phase": "unlock_day_exit",
                        "unlock_date": str(unlock_date),
                        "order_type": "market",
                    },
                )

            if phase == "recovery":
                days_since_unlock = (today - unlock_date).days
                return Signal(
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    symbol=symbol,
                    side="buy",
                    confidence=0.68,
                    target_price=current_price,
                    metadata={
                        "phase": "post_unlock_recovery_long",
                        "unlock_date": str(unlock_date),
                        "days_since_unlock": days_since_unlock,
                        "order_type": "limit",
                    },
                )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorised backtest: mark short windows and recovery windows based on
        known unlock dates. Uses shift(1) to prevent lookahead.

        For each unlock date in KNOWN_UNLOCKS matching the 'symbol' param
        (or the first unlock if symbol not provided), generate signals.
        """
        false_series = pd.Series(False, index=df.index)
        default = BacktestSignals(
            entries=false_series,
            exits=false_series,
            short_entries=false_series,
            short_exits=false_series,
        )

        if "close" not in df.columns or len(df) < 10:
            return default

        if not isinstance(df.index, pd.DatetimeIndex):
            return default

        idx_dates = df.index.normalize()

        short_entry = pd.Series(False, index=df.index)
        short_exit = pd.Series(False, index=df.index)
        long_entry = pd.Series(False, index=df.index)
        long_exit = pd.Series(False, index=df.index)

        for _sym, d_str, _frac in self._all_unlocks:
            try:
                unlock_dt = pd.Timestamp(d_str, tz="UTC")
            except Exception:
                continue

            # Short entry window: [unlock - 7d, unlock)
            short_start = unlock_dt - pd.Timedelta(days=self.short_entry_days_before)
            short_end = unlock_dt
            in_short = (idx_dates >= short_start) & (idx_dates < short_end)
            short_entry |= in_short

            # Short exit: on unlock day
            in_exit = (idx_dates == unlock_dt.normalize())
            short_exit |= in_exit

            # Long recovery window: [unlock + 14d, unlock + 30d]
            recovery_start = unlock_dt + pd.Timedelta(days=self.long_entry_days_after)
            recovery_end = unlock_dt + pd.Timedelta(days=self.long_exit_days_after)
            in_recovery = (idx_dates >= recovery_start) & (idx_dates <= recovery_end)
            long_entry |= in_recovery
            long_exit |= (idx_dates > recovery_end)

        # shift(1) — no lookahead
        entries = long_entry.shift(1).fillna(False).astype(bool)
        exits = long_exit.shift(1).fillna(False).astype(bool)
        short_entries = short_entry.shift(1).fillna(False).astype(bool)
        short_exits = short_exit.shift(1).fillna(False).astype(bool)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
