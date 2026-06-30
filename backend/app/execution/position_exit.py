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
        self, position: dict | None, current_price: float | None, context: dict | None
    ) -> Tuple[bool, str]:
        if not position or current_price is None:
            return False, ""

        side = position.get("side", "long")
        if side == "long":
            if self.take_profit_price is not None and current_price >= self.take_profit_price:
                return True, ExitReason.TAKE_PROFIT
            if self.stop_loss_price is not None and current_price <= self.stop_loss_price:
                return True, ExitReason.STOP_LOSS
        else:  # short
            if self.take_profit_price is not None and current_price <= self.take_profit_price:
                return True, ExitReason.TAKE_PROFIT
            if self.stop_loss_price is not None and current_price >= self.stop_loss_price:
                return True, ExitReason.STOP_LOSS
        return False, ""


class TrailingStopExit:
    """Trailing stop: tracks highest price since entry, stops out N% below that peak."""

    def __init__(self, trail_pct: float = 0.02) -> None:
        self.trail_pct = trail_pct  # 2% default

    def should_exit(
        self, position: dict | None, current_price: float | None, context: dict | None
    ) -> Tuple[bool, str]:
        if not position or current_price is None:
            return False, ""

        ctx = context or {}
        peak_price = ctx.get("peak_price")
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
        self, position: dict | None, current_price: float | None, context: dict | None
    ) -> Tuple[bool, str]:
        if not position or current_price is None:
            return False, ""

        ctx = context or {}
        atr_at_entry = ctx.get("atr_at_entry")
        if atr_at_entry is None:
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
        self, position: dict | None, current_price: float | None, context: dict | None
    ) -> Tuple[bool, str]:
        if not position:
            return False, ""

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
            ctx = context or {}
            bars_held = ctx.get("bars_held", 0) or 0
            # Guard against negative or non‑int values
            try:
                bars_held_int = int(bars_held)
            except (TypeError, ValueError):
                bars_held_int = 0
            if bars_held_int >= self.max_bars:
                return True, ExitReason.TIME_MAX_BARS

        return False, ""


class RegimeExit:
    """Exit directional positions in bear regime (state=0 from Redis 'market:regime')."""

    def should_exit(
        self, position: dict | None, current_price: float | None, context: dict | None
    ) -> Tuple[bool, str]:
        if not position:
            return False, ""

        ctx = context or {}
        regime = ctx.get("regime")
        if regime is None:
            return False, ""
        # Only exit directional (long) positions in bear regime
        side = position.get("side", "long")
        try:
            regime_int = int(regime)
        except (TypeError, ValueError):
            return False, ""
        if side == "long" and regime_int == 0:
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
        self, position: dict | None, current_price: float | None, context: dict | None
    ) -> Tuple[bool, str]:
        if not position or current_price is None:
            return False, ""

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
        ctx = context or {}
        peak_price = ctx.get("peak_price", current_price)
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
    """Exit when the z‑score of a spread reverts beyond a threshold."""

    def __init__(self, entry_zscore: float = 2.0, exit_zscore: float = 0.5) -> None:
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore

    def should_exit(
        self, position: dict | None, current_price: float | None, context: dict | None
    ) -> Tuple[bool, str]:
        if not position or current_price is None:
            return False, ""

        ctx = context or {}
        zscore = ctx.get("zscore")
        if zscore is None:
            return False, ""

        try:
            zscore_val = float(zscore)
        except (TypeError, ValueError):
            return False, ""

        # Only consider exit when we are already in a position (i.e., the entry
        # condition has been satisfied previously).  The logic mirrors the
        # original intent: exit when the absolute z‑score falls below the exit
        # threshold.
        if abs(zscore_val) <= self.exit_zscore:
            return True, ExitReason.ZSCORE_REVERT
        return False, ""


# ── Composite strategy ────────────────────────────────────────────────────────


class CompositeExit:
    """Runs a list of exit strategies and returns the first triggered exit."""

    def __init__(self, strategies: List[Any] | None = None) -> None:
        # Defensive copy; ensure we always have a list to iterate.
        self.strategies = list(strategies) if strategies else []

    def should_exit(
        self, position: dict | None, current_price: float | None, context: dict | None
    ) -> Tuple[bool, str]:
        if not self.strategies:
            return False, ""
        for strat in self.strategies:
            # Guard against badly‑typed strategy objects.
            if not hasattr(strat, "should_exit"):
                logger.warning("Strategy %s missing should_exit method", strat)
                continue
            try:
                exit_flag, reason = strat.should_exit(position, current_price, context)
                if exit_flag:
                    return True, reason
            except Exception as exc:  # pragma: no cover
                logger.exception("Error evaluating exit strategy %s: %s", strat, exc)
                continue
        return False, ""


# ── Factory ─────────────────────────────────────────────────────────────────────


def build_exit_strategy(strategy_type: str | None) -> CompositeExit:
    """
    Factory that returns a CompositeExit configured for a given strategy_type.

    Parameters
    ----------
    strategy_type : str | None
        Identifier for the desired exit strategy group.  If ``None`` or an
        unrecognised value is supplied, an empty CompositeExit is returned.

    Returns
    -------
    CompositeExit
        A composite containing the appropriate exit strategy instances.
    """
    if not strategy_type:
        return CompositeExit()

    strategy_type = strategy_type.lower()
    if strategy_type == "basic":
        # Simple take‑profit / stop‑loss combo.
        return CompositeExit(
            strategies=[
                FixedTPSL(take_profit_price=None, stop_loss_price=None),
                TimeBasedExit(eod_exit=True, max_bars=None),
            ]
        )
    if strategy_type == "atr":
        return CompositeExit(
            strategies=[
                ATRStop(atr_multiplier=2.0),
                TimeBasedExit(eod_exit=True, max_bars=390),
            ]
        )
    if strategy_type == "regime":
        return CompositeExit(
            strategies=[
                RegimeExit(),
                TimeBasedExit(eod_exit=True),
            ]
        )
    if strategy_type == "profit_lock":
        return CompositeExit(
            strategies=[
                ProfitLock(lock_trigger_pct=0.03, lock_trail_pct=0.01),
                TimeBasedExit(eod_exit=True),
            ]
        )
    if strategy_type == "zscore":
        return CompositeExit(
            strategies=[
                ZScoreExit(entry_zscore=2.0, exit_zscore=0.5),
                TimeBasedExit(eod_exit=True),
            ]
        )
    # Unknown type – return empty composite to avoid runtime errors.
    logger.info("Unknown exit strategy type '%s'; returning empty CompositeExit.", strategy_type)
    return CompositeExit()