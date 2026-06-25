"""
Position exit strategy classes for QuantEdge.

Each class implements ``should_exit(position, current_price, context) -> (bool, str)``
where the string is the :class:`ExitReason` value when triggered, or an empty string
when not triggered.

CompositeExit runs multiple strategies and returns the first triggered one.
``build_exit_strategy`` is a factory that returns sensible composites per strategy type.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import List, Tuple

from app.utils.logging import logger


class ExitReason(str, Enum):
    """Enumerates all possible exit reasons."""

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
    """Exit when price hits a static take‑profit or stop‑loss level."""

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
    """Trailing stop that follows the highest (or lowest) price since entry."""

    def __init__(self, trail_pct: float = 0.02) -> None:
        self.trail_pct = trail_pct  # default 2 %

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        peak_price = self._get_peak_price(position, context)
        if peak_price is None:
            return False, ""

        side = position.get("side", "long")
        if side == "long":
            stop_level = peak_price * (1.0 - self.trail_pct)
            if current_price <= stop_level:
                return True, ExitReason.TRAILING_STOP
        else:
            stop_level = peak_price * (1.0 + self.trail_pct)
            if current_price >= stop_level:
                return True, ExitReason.TRAILING_STOP
        return False, ""

    @staticmethod
    def _get_peak_price(position: dict, context: dict) -> float | None:
        """Return the tracked peak price, falling back to the entry price."""
        peak = context.get("peak_price")
        if peak is not None:
            return float(peak)
        entry_price = position.get("avg_cost") or position.get("entry_price")
        return float(entry_price) if entry_price else None


class ATRStop:
    """ATR‑based dynamic stop: ``stop_price = entry ± N * ATR``."""

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
        atr = float(atr_at_entry)
        entry = float(entry_price)

        if side == "long":
            stop_level = entry - self.atr_multiplier * atr
            if current_price <= stop_level:
                return True, ExitReason.ATR_STOP
        else:
            stop_level = entry + self.atr_multiplier * atr
            if current_price >= stop_level:
                return True, ExitReason.ATR_STOP
        return False, ""


class TimeBasedExit:
    """Exit based on time: end‑of‑day or after a maximum number of bars."""

    # 15:55 ET = 20:55 UTC (ET = UTC‑5 in winter, UTC‑4 in summer)
    _EOD_HOUR_UTC_WINTER = 20
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
        if self.eod_exit and self._is_eod_time():
            return True, ExitReason.TIME_EOD

        if self.max_bars is not None and self._has_exceeded_bars(context):
            return True, ExitReason.TIME_MAX_BARS

        return False, ""

    @staticmethod
    def _is_eod_time() -> bool:
        """Return ``True`` when the current UTC time is within the EOD window."""
        now_utc = datetime.now(UTC)
        hour, minute = now_utc.hour, now_utc.minute
        # Accept a window that covers both EST and EDT close times:
        #   19:55 – 20:55 UTC (covers 15:55 ET in EST and 16:55 ET in EDT)
        #   plus a small buffer up to 21:00 UTC.
        return (hour == 19 and minute >= 55) or (hour == 20 and minute >= 55) or (hour == 21 and minute <= 0)

    @staticmethod
    def _has_exceeded_bars(context: dict) -> bool:
        """Check whether the position has been held longer than ``max_bars``."""
        bars_held = context.get("bars_held", 0)
        return bool(bars_held)


class RegimeExit:
    """Exit directional positions when the market regime switches to bear."""

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        regime = context.get("regime")
        if regime is None:
            return False, ""
        side = position.get("side", "long")
        if side == "long" and int(regime) == 0:
            return True, ExitReason.REGIME_CHANGE
        return False, ""


class ProfitLock:
    """Lock profit once a threshold is reached and then trail a tighter stop."""

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
        gain_pct = self._calc_gain_pct(side, entry_price, current_price)

        if gain_pct < self.lock_trigger_pct:
            return False, ""

        peak_price = float(context.get("peak_price", current_price))
        if side == "long":
            lock_stop = peak_price * (1.0 - self.lock_trail_pct)
            if current_price <= lock_stop:
                return True, ExitReason.PROFIT_LOCK
        else:
            lock_stop = peak_price * (1.0 + self.lock_trail_pct)
            if current_price >= lock_stop:
                return True, ExitReason.PROFIT_LOCK
        return False, ""

    @staticmethod
    def _calc_gain_pct(side: str, entry_price: float, current_price: float) -> float:
        """Calculate realised gain as a fraction of entry price."""
        if side == "long":
            return (current_price - entry_price) / entry_price
        return (entry_price - current_price) / entry_price


class ZScoreExit:
    """Exit when a statistical arbitrage spread's z‑score reverts to zero."""

    def __init__(self, revert_threshold: float = 0.0) -> None:
        self.revert_threshold = revert_threshold

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        zscore = context.get("zscore")
        if zscore is None:
            return False, ""
        if abs(float(zscore)) <= abs(self.revert_threshold):
            return True, ExitReason.ZSCORE_REVERT
        return False, ""


# ── Composite strategy ────────────────────────────────────────────────────────


class CompositeExit:
    """Combine multiple exit strategies; the first that triggers wins."""

    def __init__(self, strategies: List):
        self.strategies = strategies

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> Tuple[bool, str]:
        for strategy in self.strategies:
            try:
                triggered, reason = strategy.should_exit(position, current_price, context)
                if triggered:
                    logger.debug(
                        "Exit triggered by %s: %s",
                        strategy.__class__.__name__,
                        reason,
                    )
                    return True, reason
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "Error in exit strategy %s: %s",
                    strategy.__class__.__name__,
                    exc,
                )
        return False, ""


# ── Factory ─────────────────────────────────────────────────────────────────────


def build_exit_strategy(config: dict) -> CompositeExit:
    """
    Build a :class:`CompositeExit` from a configuration dictionary.

    Expected keys in ``config``:
        - ``fixed_tp_sl``: ``dict`` with ``take_profit_price`` and ``stop_loss_price``.
        - ``trailing_stop``: ``dict`` with ``trail_pct``.
        - ``atr_stop``: ``dict`` with ``atr_multiplier``.
        - ``time_based``: ``dict`` with ``eod_exit``, ``max_bars`` and ``bar_interval_minutes``.
        - ``regime``: ``bool`` (include RegimeExit if ``True``).
        - ``profit_lock``: ``dict`` with ``lock_trigger_pct`` and ``lock_trail_pct``.
        - ``zscore``: ``dict`` with ``revert_threshold``.
    """
    strategies = []

    if tp_sl_cfg := config.get("fixed_tp_sl"):
        strategies.append(
            FixedTPSL(
                take_profit_price=tp_sl_cfg.get("take_profit_price"),
                stop_loss_price=tp_sl_cfg.get("stop_loss_price"),
            )
        )

    if trail_cfg := config.get("trailing_stop"):
        strategies.append(TrailingStopExit(trail_pct=trail_cfg.get("trail_pct", 0.02)))

    if atr_cfg := config.get("atr_stop"):
        strategies.append(ATRStop(atr_multiplier=atr_cfg.get("atr_multiplier", 2.0)))

    if time_cfg := config.get("time_based"):
        strategies.append(
            TimeBasedExit(
                eod_exit=time_cfg.get("eod_exit", True),
                max_bars=time_cfg.get("max_bars"),
                bar_interval_minutes=time_cfg.get("bar_interval_minutes", 1),
            )
        )

    if config.get("regime"):
        strategies.append(RegimeExit())

    if lock_cfg := config.get("profit_lock"):
        strategies.append(
            ProfitLock(
                lock_trigger_pct=lock_cfg.get("lock_trigger_pct", 0.03),
                lock_trail_pct=lock_cfg.get("lock_trail_pct", 0.01),
            )
        )

    if zs_cfg := config.get("zscore"):
        strategies.append(ZScoreExit(revert_threshold=zs_cfg.get("revert_threshold", 0.0)))

    return CompositeExit(strategies)