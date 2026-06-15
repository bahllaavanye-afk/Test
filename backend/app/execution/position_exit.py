"""
Position exit strategy classes for QuantEdge.

Each class implements should_exit(position, current_price, context) -> (bool, str)
where the string is the ExitReason value when triggered, or "" when not triggered.

CompositeExit runs multiple strategies and returns the first triggered one.
build_exit_strategy() is a factory that returns sensible composites per strategy type.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

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
    ) -> tuple[bool, str]:
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
    ) -> tuple[bool, str]:
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
    ) -> tuple[bool, str]:
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
    ) -> tuple[bool, str]:
        if self.eod_exit:
            now_utc = datetime.now(UTC)
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


class RegimeExit:
    """Exit directional positions in bear regime (state=0 from Redis 'market:regime')."""

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> tuple[bool, str]:
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
    ) -> tuple[bool, str]:
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
    """Exit when spread z-score crosses back to zero (for stat-arb / pairs trading)."""

    def __init__(self, exit_zscore: float = 0.0, timeout_bars: int = 50) -> None:
        self.exit_zscore = exit_zscore
        self.timeout_bars = timeout_bars

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> tuple[bool, str]:
        zscore = context.get("zscore")
        bars_held = context.get("bars_held", 0)

        if zscore is None:
            return False, ""

        zscore = float(zscore)
        side = position.get("side", "long")

        # Exit when z-score reverts to zero (mean reversion complete)
        if side == "long" and zscore >= self.exit_zscore:
            return True, ExitReason.ZSCORE_REVERT
        if side == "short" and zscore <= self.exit_zscore:
            return True, ExitReason.ZSCORE_REVERT

        # Timeout: close position if it hasn't reverted within timeout_bars
        if int(bars_held) >= self.timeout_bars:
            return True, ExitReason.TIME_MAX_BARS

        return False, ""


class MaxLossExit:
    """Hard cut: exit if position drawdown from entry exceeds max_loss_pct."""

    def __init__(self, max_loss_pct: float = 0.05) -> None:
        self.max_loss_pct = max_loss_pct  # 5% position-level stop

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> tuple[bool, str]:
        entry_price = position.get("avg_cost") or position.get("entry_price")
        if not entry_price:
            return False, ""

        entry_price = float(entry_price)
        side = position.get("side", "long")

        if side == "long":
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price

        if loss_pct >= self.max_loss_pct:
            return True, ExitReason.MAX_LOSS

        return False, ""


class VolatilitySpike:
    """Exit if VIX > vix_threshold while holding a directional position."""

    def __init__(self, vix_threshold: float = 30.0) -> None:
        self.vix_threshold = vix_threshold

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> tuple[bool, str]:
        vix = context.get("vix")
        if vix is None:
            return False, ""
        if float(vix) > self.vix_threshold:
            return True, ExitReason.VOLATILITY_SPIKE
        return False, ""


# ── Composite: run multiple strategies, first trigger wins ───────────────────

class CompositeExit:
    """Runs multiple exit strategies. First one to trigger wins."""

    def __init__(self, strategies: list) -> None:
        self.strategies = strategies

    def should_exit(
        self, position: dict, current_price: float, context: dict
    ) -> tuple[bool, str | None]:
        for strategy in self.strategies:
            try:
                triggered, reason = strategy.should_exit(position, current_price, context)
                if triggered:
                    return True, reason
            except Exception as exc:
                logger.warning(
                    "Exit strategy check failed",
                    strategy=type(strategy).__name__,
                    error=str(exc),
                )
        return False, None


# ── Factory: build standard exit composites per strategy type ─────────────────

def build_exit_strategy(
    strategy_type: str,
    risk_bucket: str,
    params: dict,
) -> CompositeExit:
    """
    Returns a CompositeExit configured for the strategy type:

    - arbitrage:   FixedTPSL(tight) + MaxLoss(2%) + ZScoreExit + TimeBasedExit(EOD)
    - directional: FixedTPSL + ATRStop(2x) + TrailingStop + ProfitLock + RegimeExit
    - options:     TimeBasedExit(21DTE) + FixedTPSL(50% profit target)
    - crypto_arb:  MaxLoss(1%) + ZScoreExit + TimeBasedExit(max_bars=20)
    """
    stop_loss = params.get("stop_loss")
    take_profit = params.get("take_profit")

    if strategy_type in ("arbitrage",) or risk_bucket == "arbitrage":
        return CompositeExit([
            FixedTPSL(
                take_profit_price=take_profit,
                stop_loss_price=stop_loss,
            ),
            MaxLossExit(max_loss_pct=params.get("max_loss_pct", 0.02)),
            ZScoreExit(
                exit_zscore=params.get("exit_zscore", 0.0),
                timeout_bars=params.get("timeout_bars", 50),
            ),
            TimeBasedExit(eod_exit=True),
        ])

    elif strategy_type == "crypto_arb":
        return CompositeExit([
            MaxLossExit(max_loss_pct=params.get("max_loss_pct", 0.01)),
            ZScoreExit(
                exit_zscore=params.get("exit_zscore", 0.0),
                timeout_bars=params.get("timeout_bars", 20),
            ),
            TimeBasedExit(
                eod_exit=False,
                max_bars=params.get("max_bars", 20),
            ),
        ])

    elif strategy_type == "options":
        return CompositeExit([
            TimeBasedExit(
                eod_exit=False,
                max_bars=params.get("max_bars_dte", 21),
                bar_interval_minutes=60 * 24,  # daily bars
            ),
            FixedTPSL(
                take_profit_price=take_profit,
                stop_loss_price=stop_loss,
            ),
            MaxLossExit(max_loss_pct=params.get("max_loss_pct", 1.0)),  # options: 100% premium loss
        ])

    else:
        # directional (default)
        return CompositeExit([
            FixedTPSL(
                take_profit_price=take_profit,
                stop_loss_price=stop_loss,
            ),
            ATRStop(atr_multiplier=params.get("atr_multiplier", 2.0)),
            TrailingStopExit(trail_pct=params.get("trail_pct", 0.02)),
            ProfitLock(
                lock_trigger_pct=params.get("lock_trigger_pct", 0.03),
                lock_trail_pct=params.get("lock_trail_pct", 0.01),
            ),
            RegimeExit(),
            MaxLossExit(max_loss_pct=params.get("max_loss_pct", 0.05)),
            VolatilitySpike(vix_threshold=params.get("vix_threshold", 30.0)),
        ])
