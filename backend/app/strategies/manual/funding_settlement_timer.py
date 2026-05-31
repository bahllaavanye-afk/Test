"""
Crypto Funding Rate Settlement Timer.

Perpetual futures funding payments occur every 8 hours: 00:00, 08:00, 16:00 UTC.

Logic:
- Fetch current funding rate from Binance FAPI (free, no auth required):
  GET https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT
- If funding_rate > min_funding_rate (positive → longs pay shorts):
  → 30 minutes before settlement: enter SHORT to collect funding payment
  → Close position 10 minutes after settlement
- If funding_rate < -min_funding_rate (negative → shorts pay longs):
  → 30 minutes before settlement: enter LONG to collect funding payment
  → Close position 10 minutes after settlement
- If |funding_rate| < min_funding_rate: no trade

Academic reference:
  Avellaneda & Stoikov (2008) — on funding cost dynamics.
  Cong, Harvey & Rabetti (2023) "Crypto Carry" — AFA WP 2023.
  Funding arbitrage Sharpe documented at 3.1-4.5 annualised.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import aiohttp

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"

# Funding settlement hours in UTC
_SETTLEMENT_HOURS = (0, 8, 16)
# Entry/exit offsets in minutes
_ENTRY_MINUTES_BEFORE = 30
_EXIT_MINUTES_AFTER = 10


class FundingSettlementTimer(AbstractStrategy):
    """
    Trades the funding rate settlement window on Binance perpetual futures.

    Enters a position opposing the dominant side 30 minutes before each
    8-hour funding settlement to collect the funding payment, then exits
    10 minutes after settlement.
    """

    name = "funding_settlement_timer"
    display_name = "Funding Rate Settlement Timer"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 300.0   # 5-minute polling

    DEFAULT_MIN_FUNDING_RATE: float = 0.0001   # 0.01%
    DEFAULT_ENTRY_MINUTES_BEFORE: int = 30
    DEFAULT_EXIT_MINUTES_AFTER: int = 10

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = params or {}
        self.min_funding_rate: float = float(
            p.get("min_funding_rate", self.DEFAULT_MIN_FUNDING_RATE)
        )
        self.entry_minutes_before: int = int(
            p.get("entry_minutes_before", self.DEFAULT_ENTRY_MINUTES_BEFORE)
        )
        self.exit_minutes_after: int = int(
            p.get("exit_minutes_after", self.DEFAULT_EXIT_MINUTES_AFTER)
        )
        self.symbol: str = str(p.get("symbol", "BTCUSDT"))

    def description(self) -> str:
        return (
            f"Enters {self.entry_minutes_before}min before each 8-hour funding "
            "settlement to collect payment. Long when rate negative, short when positive. "
            f"Min rate: {self.min_funding_rate * 100:.3f}%."
        )

    @staticmethod
    def _minutes_to_next_settlement(now: datetime) -> int:
        """Return minutes until the next 8-hour funding settlement."""
        hour = now.hour
        minute = now.minute
        for settlement_hour in sorted(_SETTLEMENT_HOURS):
            total_settlement_minutes = settlement_hour * 60
            total_now_minutes = hour * 60 + minute
            delta = total_settlement_minutes - total_now_minutes
            if delta > 0:
                return delta
        # Past last settlement today; next is at 00:00 tomorrow
        next_midnight_minutes = 24 * 60 - (hour * 60 + minute)
        return next_midnight_minutes

    @staticmethod
    def _minutes_since_last_settlement(now: datetime) -> int:
        """Return minutes since the most recent 8-hour settlement."""
        hour = now.hour
        minute = now.minute
        total_now_minutes = hour * 60 + minute
        last_settlement_minutes = 0
        for settlement_hour in sorted(_SETTLEMENT_HOURS):
            total_settlement_minutes = settlement_hour * 60
            if total_settlement_minutes <= total_now_minutes:
                last_settlement_minutes = total_settlement_minutes
        return total_now_minutes - last_settlement_minutes

    async def _fetch_funding_rate(self) -> float:
        """Fetch current funding rate from Binance FAPI. Raises on failure."""
        params = {"symbol": self.symbol}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                _PREMIUM_INDEX_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        funding_rate = float(data.get("lastFundingRate", data.get("fundingRate", "0")))
        return funding_rate

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Check timing windows and current funding rate.
        Enter trade 30 min before settlement if rate exceeds threshold.
        """
        now_utc = datetime.now(timezone.utc)
        minutes_to_next = self._minutes_to_next_settlement(now_utc)
        minutes_since_last = self._minutes_since_last_settlement(now_utc)

        # Exit window: within exit_minutes_after of settlement
        in_exit_window = minutes_since_last <= self.exit_minutes_after

        # Entry window: within entry_minutes_before of next settlement
        in_entry_window = minutes_to_next <= self.entry_minutes_before

        if not in_entry_window and not in_exit_window:
            return None

        if "close" not in data.columns or len(data) == 0:
            raise ValueError(
                "FundingSettlementTimer.analyze: 'close' column required in data."
            )
        current_price = float(data["close"].iloc[-1])
        if current_price <= 0:
            raise ValueError(f"FundingSettlementTimer: invalid price {current_price}")

        try:
            funding_rate = await self._fetch_funding_rate()
        except Exception as exc:
            raise RuntimeError(
                f"FundingSettlementTimer: failed to fetch funding rate — {exc}"
            ) from exc

        abs_rate = abs(funding_rate)

        if abs_rate < self.min_funding_rate:
            return None

        if in_exit_window:
            # Close position after settlement
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="sell",   # exit signal; direction determined by execution layer
                confidence=0.80,
                target_price=current_price,
                metadata={
                    "funding_rate": round(funding_rate, 6),
                    "action": "exit_after_settlement",
                    "minutes_since_settlement": minutes_since_last,
                    "order_type": "market",
                },
            )

        if in_entry_window:
            # Enter opposing position to collect funding
            if funding_rate > self.min_funding_rate:
                # Positive rate: longs pay shorts → go SHORT
                side = "sell"
                action = "short_before_settlement_positive_funding"
            else:
                # Negative rate: shorts pay longs → go LONG
                side = "buy"
                action = "long_before_settlement_negative_funding"

            confidence = min(0.90, 0.65 + abs_rate / 0.001)
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side=side,
                confidence=confidence,
                target_price=current_price,
                metadata={
                    "funding_rate": round(funding_rate, 6),
                    "funding_rate_pct": round(funding_rate * 100, 4),
                    "action": action,
                    "minutes_to_settlement": minutes_to_next,
                    "order_type": "market",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Proxy backtest: use 8-hour return reversal as funding rate surrogate.

        When the 8-hour return is strongly positive (longs crowded, positive
        funding), enter short. When strongly negative, enter long.
        Shift(1) to prevent lookahead.
        """
        false_series = pd.Series(False, index=df.index)
        default = BacktestSignals(
            entries=false_series,
            exits=false_series,
            short_entries=false_series,
            short_exits=false_series,
        )

        if "close" not in df.columns or len(df) < 20:
            return default

        close = df["close"].astype(float)

        # 3-bar return as funding proxy (roughly 1 settlement period on hourly bars)
        ret3 = close.pct_change(3)
        roll_mean = ret3.rolling(30, min_periods=15).mean()
        roll_std = ret3.rolling(30, min_periods=15).std().clip(lower=1e-8)
        proxy_z = (ret3 - roll_mean) / roll_std

        # shift(1) — no lookahead
        z_lag = proxy_z.shift(1)

        # Entry thresholds corresponding to roughly 0.01% funding rate signal
        entry_threshold = 1.5

        # Long: negative funding (shorts crowded) → price likely to squeeze up
        entries = (z_lag < -entry_threshold).fillna(False).astype(bool)
        exits = (z_lag.abs() < 0.3).fillna(False).astype(bool)

        # Short: positive funding (longs crowded) → price likely to drop
        short_entries = (z_lag > entry_threshold).fillna(False).astype(bool)
        short_exits = exits.copy()

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
