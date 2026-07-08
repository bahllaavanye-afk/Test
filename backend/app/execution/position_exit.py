"""
Position exit strategy classes for QuantEdge.

Each class implements should_exit(position, current_price, context) -> (bool, str)
where the string is the ExitReason value when triggered, or "" when not triggered.

CompositeExit runs multiple strategies and returns the first triggered one.
build_exit_strategy() is a factory that returns sensible composites per strategy type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Tuple

from app.utils.logging import logger


class ExitReason(str, Enum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_EOD = "time_eod"
    TIME_MAX_BARS = "time_max_bars"
    ATR_STOP = "atr_stop"
    REGIME_CHANGE = "regime_change"
    PROFIT_LOCK = "profit_lock"         # ratchet stop once up X%
    ZSCORE_REVERT = "zscore_revert"     # stat-arb mean reversion
    MAX_LOSS = "max_loss"               # position-level drawdown cap
    VOLATILITY_SPIKE = "vol_spike"      # exit if VIX spikes > threshold


# ── Individual exit strategies ────────────────────────────────────────────────

class FixedTPSL:
    """Exit when price hits take_profit or stop_loss set at entry."""

    def __init__(
        self,
        take_profit_price: float | None,
        stop_loss_price: float | None,
    ) -> None:
        self.take_profit_price = take_profit_price
        self.stop_loss_price = stop_loss_price

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        try:
            side = position.get("side", "long")
            if side == "long":
                if self.take_profit_price and current_price >= self.take_profit_price:
                    return True, ExitReason.TAKE_PROFIT
                if self.stop_loss_price and current_price <= self.stop_loss_price:
                    return True, ExitReason.STOP_LOSS
            else:  # short
                if self.take_profit_price and current_price <= self.take_profit_price:
                    return True, ExitReason.TAKE_PROFIT
                if self.stop_loss_price and current_price >= self.stop_loss_price:
                    return True, ExitReason.STOP_LOSS
            return False, ""
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                "FixedTPSL.should_exit encountered an error",
                extra={"position": position, "current_price": current_price, "context": context},
                exc_info=e,
            )
            return False, ""


class TrailingStopExit:
    """Trailing stop: tracks highest price since entry, stops out N% below that peak."""

    def __init__(self, trail_pct: float = 0.02) -> None:
        self.trail_pct = trail_pct  # 2% default

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        try:
            peak_price = context.get("peak_price")
            if peak_price is None:
                # Fall back to entry price if peak not tracked yet
                peak_price = position.get("avg_cost") or position.get("entry_price")
            if not peak_price:
                return False, ""

            side = position.get("side", "long")
            if side == "long":
                stop_level = float(peak_price) * (1.0 - self.trail_pct)
                if current_price <= stop_level:
                    return True, ExitReason.TRAILING_STOP
            else:
                # For short, trailing stop tracks the lowest price
                stop_level = float(peak_price) * (1.0 + self.trail_pct)
                if current_price >= stop_level:
                    return True, ExitReason.TRAILING_STOP
            return False, ""
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                "TrailingStopExit.should_exit encountered an error",
                extra={"position": position, "current_price": current_price, "context": context},
                exc_info=e,
            )
            return False, ""


class ATRStop:
    """ATR-based dynamic stop: stop_price = entry - N * ATR at entry time."""

    def __init__(self, atr_multiplier: float = 2.0) -> None:
        self.atr_multiplier = atr_multiplier

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        try:
            atr_at_entry = context.get("atr_at_entry")
            if not atr_at_entry:
                return False, ""

            entry_price = position.get("avg_cost") or position.get("entry_price")
            if not entry_price:
                return False, ""

            side = position.get("side", "long")
            atr_at_entry = float(atr_at_entry)
            entry_price = float(entry_price)

            if side == "long":
                stop_level = entry_price - self.atr_multiplier * atr_at_entry
                if current_price <= stop_level:
                    return True, ExitReason.ATR_STOP
            else:
                stop_level = entry_price + self.atr_multiplier * atr_at_entry
                if current_price >= stop_level:
                    return True, ExitReason.ATR_STOP
            return False, ""
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                "ATRStop.should_exit encountered an error",
                extra={"position": position, "current_price": current_price, "context": context},
                exc_info=e,
            )
            return False, ""


class TimeBasedExit:
    """Exit at EOD (15:55 ET for equities), or after max_bbars elapsed."""

    # 15:55 ET = 20:55 UTC (ET = UTC-5 in winter, UTC-4 in summer)
    _EOD_HOUR_UTC_WINTER = 20   # 15:55 ET (EST = UTC-5)
    _EOD_MINUTE = 55

    def __init__(
        self,
        eod_exit: bool = True,
        max_bars: int | None = None,
        bar_interval_minutes: int = 1,
    ) -> None:
        self.eod_exit = eod_exit
        self.max_bars = max_bars
        self.bar_interval_minutes = bar_interval_minutes

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        try:
            if self.eod_exit:
                now_utc = datetime.now(timezone.utc)
                # Approximate EOD check: 20:55 UTC covers both EST (UTC-5) and EDT (UTC-4)
                # In EDT the market close (16:00 ET) is 20:00 UTC — use 19:55 for that case
                # We use a range: fire if 19:55 <= now_utc.time <= 21:00 UTC
                hour = now_utc.hour
                minute = now_utc.minute
                if (hour == 19 and minute >= 55) or (hour == 20 and minute >= 55):
                    return True, ExitReason.TIME_EOD

            if self.max_bars is not None:
                bars_held = context.get("bars_held", 0)
                if bars_held >= self.max_bars:
                    return True, ExitReason.TIME_MAX_BARS

            return False, ""
        except Exception as e:
            logger.error(
                "TimeBasedExit.should_exit encountered an unexpected error",
                extra={"position": position, "current_price": current_price, "context": context},
                exc_info=e,
            )
            return False, ""


class RegimeExit:
    """Exit directional positions in bear regime (state=0 from Redis 'market:regime')."""

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        try:
            regime = context.get("regime")
            if regime is None:
                return False, ""
            # Only exit directional (long) positions in bear regime
            side = position.get("side", "long")
            if side == "long" and int(regime) == 0:
                return True, ExitReason.REGIME_CHANGE
            return False, ""
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                "RegimeExit.should_exit encountered an error",
                extra={"position": position, "current_price": current_price, "context": context},
                exc_info=e,
            )
            return False, ""


class ProfitLock:
    """Once position is up lock_trigger_pct, activate trailing stop at lock_trail_pct below peak."""

    def __init__(
        self,
        lock_trigger_pct: float = 0.03,
        lock_trail_pct: float = 0.01,
    ) -> None:
        self.lock_trigger_pct = lock_trigger_pct
        self.lock_trail_pct = lock_trail_pct

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        try:
            entry_price = position.get("avg_cost") or position.get("entry_price")
            if not entry_price:
                return False, ""

            entry_price = float(entry_price)
            side = position.get("side", "long")

            if side == "long":
                gain_pct = (current_price - entry_price) / entry_price
            else:
                gain_pct = (entry_price - current_price) / entry_price

            if gain_pct < self.lock_trigger_pct:
                # Not yet in profit-lock territory
                return False, ""

            # Profit lock activated — use peak_price to determine trailing stop
            peak_price = context.get("peak_price", current_price)
            peak_price = float(peak_price)

            if side == "long":
                lock_stop = peak_price * (1.0 - self.lock_trail_pct)
                if current_price <= lock_stop:
                    return True, ExitReason.PROFIT_LOCK
            else:
                lock_stop = peak_price * (1.0 + self.lock_trail_pct)
                if current_price >= lock_stop:
                    return True, ExitReason.PROFIT_LOCK

            return False, ""
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                "ProfitLock.should_exit encountered an error",
                extra={"position": position, "current_price": current_price, "context": context},
                exc_info=e,
            )
            return False, ""


class ZScoreExit:
    """Exit when spread 
# ... (truncated for brevity)"""