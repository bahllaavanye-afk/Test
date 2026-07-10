"""
Position exit strategy classes for QuantEdge.

Each class implements ``should_exit(position, current_price, context) -> (bool, str)``
where the string is the :class:`ExitReason` value when triggered, or an empty string
when not triggered.

The module provides:

* Individual exit strategy classes.
* :class:`CompositeExit` – runs multiple strategies and returns the first triggered
  exit reason.
* :func:`build_exit_strategy` – factory that builds sensible composites per strategy
  type.

All classes are lightweight and side‑effect free; they rely solely on the supplied
``position`` and ``context`` dictionaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Tuple, Callable, Optional

from app.utils.logging import logger


class ExitReason(str, Enum):
    """Enumerates the possible exit reasons."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_EOD = "time_eod"
    TIME_MAX_BARS = "time_max_bars"
    ATR_STOP = "atr_stop"
    REGIME_CHANGE = "regime_change"
    PROFIT_LOCK = "profit_lock"         # ratchet stop once up X%
    ZSCORE_REVERT = "zscore_revert"     # stat‑arb mean reversion
    MAX_LOSS = "max_loss"               # position‑level drawdown cap
    VOLATILITY_SPIKE = "vol_spike"      # exit if VIX spikes > threshold


# ── Individual exit strategies ────────────────────────────────────────────────


class FixedTPSL:
    """Exit when price hits a fixed take‑profit or stop‑loss level set at entry.

    Parameters
    ----------
    take_profit_price: float | None
        The absolute price at which a long position should be taken profit.
    stop_loss_price: float | None
        The absolute price at which a position should be stopped out.
    """

    def __init__(
        self,
        take_profit_price: float | None,
        stop_loss_price: float | None,
    ) -> None:
        self.take_profit_price = take_profit_price
        self.stop_loss_price = stop_loss_price

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Determine whether the position should exit based on fixed TP/SL levels.

        Returns
        -------
        Tuple[bool, str]
            ``(True, ExitReason.TAKE_PROFIT)`` or ``(True, ExitReason.STOP_LOSS)`` if
            the respective level is breached, otherwise ``(False, "")``.
        """
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
    """Trailing stop that tracks the highest (or lowest) price since entry.

    The stop is placed ``trail_pct`` below the peak for long positions and
    ``trail_pct`` above the trough for short positions.
    """

    def __init__(self, trail_pct: float = 0.02) -> None:
        self.trail_pct = trail_pct  # 2 % default

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Return ``True`` when the price breaches the trailing stop level."""
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
    """ATR‑based dynamic stop: ``stop_price = entry ± N × ATR`` at entry time."""

    def __init__(self, atr_multiplier: float = 2.0) -> None:
        self.atr_multiplier = atr_multiplier

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Return ``True`` when price crosses the ATR‑derived stop level."""
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
    """Exit based on clock time (EOD) or maximum number of bars held."""

    # 15:55 ET = 20:55 UTC (ET = UTC‑5 in winter, UTC‑4 in summer)
    _EOD_HOUR_UTC_WINTER = 20   # 15:55 ET (EST = UTC‑5)
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
        self,
        position: Dict[str, Any],
        current_price: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Return ``True`` when either EOD time is reached or max bars elapsed."""
        if self.eod_exit:
            now_utc = datetime.now(timezone.utc)
            hour = now_utc.hour
            minute = now_utc.minute
            # Fire if 19:55 – 21:00 UTC (covers EST and EDT close times)
            if (hour == 19 and minute >= 55) or (hour == 20 and minute >= 55):
                return True, ExitReason.TIME_EOD

        if self.max_bars is not None:
            bars_held = context.get("bars_held", 0)
            if bars_held >= self.max_bars:
                return True, ExitReason.TIME_MAX_BARS

        return False, ""


class RegimeExit:
    """Exit directional (long) positions when the market regime switches to bear."""

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Return ``True`` if the regime is bearish (state = 0) for a long position."""
        regime = context.get("regime")
        if regime is None:
            return False, ""
        side = position.get("side", "long")
        if side == "long" and int(regime) == 0:
            return True, ExitReason.REGIME_CHANGE
        return False, ""


class ProfitLock:
    """Ratchet stop that activates after a profit threshold is reached.

    Once the position is up ``lock_trigger_pct`` the trailing stop is set
    ``lock_trail_pct`` below (or above for shorts) the observed peak price.
    """

    def __init__(
        self,
        lock_trigger_pct: float = 0.03,
        lock_trail_pct: float = 0.01,
    ) -> None:
        self.lock_trigger_pct = lock_trigger_pct
        self.lock_trail_pct = lock_trail_pct

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Return ``True`` when the trailing lock stop is breached."""
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
            # Not yet in profit‑lock territory
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
    """Exit when a statistical arbitrage spread reverts past a z‑score threshold."""

    def __init__(self, zscore_threshold: float = 2.0) -> None:
        self.zscore_threshold = zscore_threshold

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Return ``True`` when the absolute z‑score exceeds the configured limit."""
        zscore = context.get("zscore")
        if zscore is None:
            return False, ""
        if abs(float(zscore)) >= self.zscore_threshold:
            return True, ExitReason.ZSCORE_REVERT
        return False, ""


# ── Composite exit strategy ────────────────────────────────────────────────────


class CompositeExit:
    """Runs a collection of exit strategies and returns the first triggered one.

    Parameters
    ----------
    strategies: List[Callable[[Dict[str, Any], float, Dict[str, Any]], Tuple[bool, str]]]
        A list of objects that implement ``should_exit`` with the same signature.
    """

    def __init__(
        self,
        strategies: List[Callable[[Dict[str, Any], float, Dict[str, Any]], Tuple[bool, str]]],
    ) -> None:
        self.strategies = strategies

    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Iterate over the configured strategies and return the first exit signal.

        If none of the strategies trigger, ``(False, "")`` is returned.
        """
        for strat in self.strategies:
            try:
                exit_flag, reason = strat.should_exit(position, current_price, context)
                if exit_flag:
                    return True, reason
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "CompositeExit: strategy %s raised an exception: %s",
                    getattr(strat, "__class__", str(strat)),
                    exc,
                )
        return False, ""


# ── Factory ───────────────────────────────────────────────────────────────────────


def build_exit_strategy(
    strategy_type: str,
    **kwargs: Any,
) -> CompositeExit:
    """Factory that builds a :class:`CompositeExit` based on a strategy name.

    Supported ``strategy_type`` values:

    * ``"fixed_tp_sl"`` – :class:`FixedTPSL` with ``take_profit_price`` and
      ``stop_loss_price`` supplied via ``kwargs``.
    * ``"trailing_stop"`` – :class:`TrailingStopExit` (optional ``trail_pct``).
    * ``"atr_stop"`` – :class:`ATRStop` (optional ``atr_multiplier``).
    * ``"time_based"`` – :class:`TimeBasedExit` (optional ``eod_exit``,
      ``max_bars`` and ``bar_interval_minutes``).
    * ``"regime"`` – :class:`RegimeExit`.
    * ``"profit_lock"`` – :class:`ProfitLock` (optional ``lock_trigger_pct``,
      ``lock_trail_pct``).
    * ``"zscore"`` – :class:`ZScoreExit` (optional ``zscore_threshold``).

    Additional strategies can be combined by passing a list to the ``extra_strategies``
    keyword argument.

    Returns
    -------
    CompositeExit
        An instance ready to be used by the execution engine.
    """
    extra_strategies: List[Callable[[Dict[str, Any], float, Dict[str, Any]], Tuple[bool, str]]] = kwargs.pop(
        "extra_strategies", []
    )

    if strategy_type == "fixed_tp_sl":
        strat = FixedTPSL(
            take_profit_price=kwargs.get("take_profit_price"),
            stop_loss_price=kwargs.get("stop_loss_price"),
        )
    elif strategy_type == "trailing_stop":
        strat = TrailingStopExit(trail_pct=kwargs.get("trail_pct", 0.02))
    elif strategy_type == "atr_stop":
        strat = ATRStop(atr_multiplier=kwargs.get("atr_multiplier", 2.0))
    elif strategy_type == "time_based":
        strat = TimeBasedExit(
            eod_exit=kwargs.get("eod_exit", True),
            max_bars=kwargs.get("max_bars"),
            bar_interval_minutes=kwargs.get("bar_interval_minutes", 1),
        )
    elif strategy_type == "regime":
        strat = RegimeExit()
    elif strategy_type == "profit_lock":
        strat = ProfitLock(
            lock_trigger_pct=kwargs.get("lock_trigger_pct", 0.03),
            lock_trail_pct=kwargs.get("lock_trail_pct", 0.01),
        )
    elif strategy_type == "zscore":
        strat = ZScoreExit(zscore_threshold=kwargs.get("zscore_threshold", 2.0))
    else:
        raise ValueError(f"Unsupported exit strategy type: {strategy_type}")

    # Combine the primary strategy with any extra ones supplied by the caller.
    strategies = [strat] + extra_strategies
    return CompositeExit(strategies)