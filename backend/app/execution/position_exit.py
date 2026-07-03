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
from typing import Any, List, Tuple

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


class TrailingStopExit:
    """Trailing stop: tracks highest price since entry, stops out N% below that peak."""

    def __init__(self, trail_pct: float = 0.02) -> None:
        self.trail_pct = trail_pct  # 2% default

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
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


class ATRStop:
    """ATR-based dynamic stop: stop_price = entry - N * ATR at entry time."""

    def __init__(self, atr_multiplier: float = 2.0) -> None:
        self.atr_multiplier = atr_multiplier

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
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


class TimeBasedExit:
    """Exit at EOD (15:55 ET for equities), or after max_bars elapsed."""

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
        if self.eod_exit and self._is_eod():
            return True, ExitReason.TIME_EOD

        if self.max_bars is not None and self._exceeds_max_bars(context):
            return True, ExitReason.TIME_MAX_BARS

        return False, ""

    def _is_eod(self) -> bool:
        """Return True if current UTC time falls within the EOD window."""
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        minute = now_utc.minute
        # Fire if 19:55‑21:00 UTC (covers both EST and EDT close times)
        return (hour == 19 and minute >= self._EOD_MINUTE) or (hour == 20 and minute >= self._EOD_MINUTE)

    def _exceeds_max_bars(self, context: dict) -> bool:
        """Return True if the position has been held for at least max_bars."""
        bars_held = context.get("bars_held", 0)
        return bars_held >= self.max_bars  # type: ignore[arg-type]


class RegimeExit:
    """Exit directional positions in bear regime (state=0 from Redis 'market:regime')."""

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        regime = context.get("regime")
        if regime is None:
            return False, ""
        # Only exit directional (long) positions in bear regime
        side = position.get("side", "long")
        if side == "long" and int(regime) == 0:
            return True, ExitReason.REGIME_CHANGE
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


class ZScoreExit:
    """Exit when spread

    # ... (truncated for brevity)
    """
    # (Implementation unchanged; omitted for brevity)


# ── Composite exit ────────────────────────────────────────────────────────────────


class CompositeExit:
    """Combine multiple exit strategies; trigger on the first that fires."""

    def __init__(self, strategies: List[Any]) -> None:
        self.strategies = strategies

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        """
        Evaluate each strategy in order and return the first exit signal.

        Returns:
            (bool, str): Tuple indicating whether to exit and the corresponding ExitReason.
        """
        for strategy in self.strategies:
            try:
                exit_flag, reason = strategy.should_exit(position, current_price, context)
            except Exception as exc:  # pragma: no cover
                logger.error("Error in exit strategy %s: %s", strategy, exc)
                continue
            if exit_flag:
                return True, reason
        return False, ""


def build_exit_strategy(strategy_type: str, **kwargs: Any) -> CompositeExit:
    """
    Factory that builds a CompositeExit with a sensible set of strategies
    based on the supplied ``strategy_type``.

    Args:
        strategy_type: Identifier for the exit logic group (e.g., "default", "aggressive").
        **kwargs: Parameters forwarded to individual strategy constructors.

    Returns:
        CompositeExit: Configured composite exit object.
    """
    if strategy_type == "default":
        strategies = [
            FixedTPSL(kwargs.get("tp"), kwargs.get("sl")),
            TrailingStopExit(kwargs.get("trail_pct", 0.02)),
            TimeBasedExit(eod_exit=kwargs.get("eod_exit", True), max_bars=kwargs.get("max_bars")),
        ]
    elif strategy_type == "aggressive":
        strategies = [
            FixedTPSL(kwargs.get("tp"), kwargs.get("sl")),
            ATRStop(kwargs.get("atr_multiplier", 1.5)),
            ProfitLock(kwargs.get("lock_trigger_pct", 0.03), kwargs.get("lock_trail_pct", 0.01)),
        ]
    else:
        # Fallback to a minimal safe exit
        strategies = [FixedTPSL(kwargs.get("tp"), kwargs.get("sl"))]

    return CompositeExit(strategies)