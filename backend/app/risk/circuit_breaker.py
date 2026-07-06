"""Drawdown-based circuit breakers — halt trading at configurable thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Optional

from app.utils.logging import logger


class BreakerState(str, Enum):
    NORMAL = "normal"
    HALTED = "halted"


@dataclass
class CircuitBreaker:
    """
    Circuit breaker that monitors equity drawdown and halts trading when thresholds are breached.

    The breaker can be configured to require a consecutive number of drawdown breaches
    (confirmation_period) before entering the HALTED state, reducing false positives.
    It can also automatically recover when the drawdown falls below a recovery threshold.
    Additional filters tighten entry conditions and improve exit logic.
    """

    # Core configuration
    name: str
    max_drawdown_pct: float                     # e.g. 0.10 = 10%

    # Optional tightening filters
    min_drawdown_abs: float = 0.0               # absolute equity drop required to trigger a breach
    hysteresis_pct: float = 0.0                 # drawdown margin below max_drawdown_pct to reset breach counter

    # Recovery / cooldown configuration
    recovery_drawdown_pct: float = 0.0          # drawdown pct below which the breaker auto‑resets
    cooldown_seconds: int = 0                  # mandatory quiet period after a reset

    # Operational state
    peak_equity: float = 0.0
    current_equity: float = 0.0
    state: BreakerState = BreakerState.NORMAL
    halted_at: Optional[datetime] = None
    halt_reasons: List[str] = field(default_factory=list)

    # Confirmation logic
    confirmation_period: int = 1                # number of consecutive breaches required to halt

    # Internal tracking (not part of the public dataclass fields)
    _breach_count: int = field(init=False, default=0)
    _cooldown_until: datetime = field(
        init=False,
        default_factory=lambda: datetime.min.replace(tzinfo=timezone.utc),
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (0.0 < self.max_drawdown_pct <= 1.0):
            raise ValueError("max_drawdown_pct must be between 0 and 1")
        if self.confirmation_period < 1:
            raise ValueError("confirmation_period must be at least 1")
        if self.min_drawdown_abs < 0.0:
            raise ValueError("min_drawdown_abs cannot be negative")
        if not (0.0 <= self.hysteresis_pct < self.max_drawdown_pct):
            raise ValueError("hysteresis_pct must be non‑negative and less than max_drawdown_pct")
        if self.recovery_drawdown_pct < 0.0:
            raise ValueError("recovery_drawdown_pct cannot be negative")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")

    def update(self, equity: float) -> bool:
        """
        Update the breaker with the latest equity snapshot.

        Returns:
            bool: True if the breaker remains in NORMAL state, False if HALTED.
        """
        if equity is None or not isinstance(equity, (int, float)):
            logger.warning(
                "Circuit breaker received invalid equity value",
                name=self.name,
                equity=equity,
            )
            return not self.is_halted

        now = datetime.now(timezone.utc)

        # Update peak and current equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.current_equity = equity

        # Enforce cooldown after a reset
        if now < self._cooldown_until:
            logger.debug(
                "Circuit breaker in cooldown period",
                name=self.name,
                cooldown_ends=self._cooldown_until,
                now=now,
            )
            return True

        # If already halted, check for possible auto‑recovery
        if self.state == BreakerState.HALTED:
            if self._should_recover():
                self.reset(equity)
                logger.info("Circuit breaker auto‑recovered", name=self.name)
                return True
            logger.debug(
                "Circuit breaker remains halted",
                name=self.name,
                drawdown=self.current_drawdown,
                recovery_threshold=self.recovery_drawdown_pct,
            )
            return False

        # Compute current drawdown
        drawdown = self.current_drawdown

        # Apply absolute drawdown filter
        absolute_drop = (self.peak_equity - self.current_equity)
        if drawdown >= self.max_drawdown_pct and absolute_drop >= self.min_drawdown_abs:
            self._breach_count += 1
            logger.debug(
                "Circuit breaker drawdown breach",
                name=self.name,
                drawdown=drawdown,
                threshold=self.max_drawdown_pct,
                absolute_drop=absolute_drop,
                min_abs=self.min_drawdown_abs,
                breach_count=self._breach_count,
            )
            if self._breach_count >= self.confirmation_period:
                self.state = BreakerState.HALTED
                self.halted_at = now
                reason = (
                    f"Drawdown {drawdown:.2%} >= threshold {self.max_drawdown_pct:.2%} "
                    f"(abs {absolute_drop:.2f}) confirmed {self._breach_count}×"
                )
                self.halt_reasons.append(reason)
                logger.error(
                    "Circuit breaker TRIPPED",
                    name=self.name,
                    drawdown=drawdown,
                    threshold=self.max_drawdown_pct,
                    reason=reason,
                )
                return False
        else:
            # Reset breach counter when drawdown falls sufficiently below the threshold
            reset_limit = self.max_drawdown_pct - self.hysteresis_pct
            if self._breach_count and drawdown < reset_limit:
                logger.debug(
                    "Circuit breaker breach counter reset due to hysteresis",
                    name=self.name,
                    previous_breach_count=self._breach_count,
                    drawdown=drawdown,
                    reset_limit=reset_limit,
                )
                self._breach_count = 0

        return True

    def _should_recover(self) -> bool:
        """
        Determine whether the breaker should automatically recover.

        Recovery occurs when the current drawdown falls below `recovery_drawdown_pct`.
        """
        if self.recovery_drawdown_pct <= 0.0:
            return False
        return self.current_drawdown <= self.recovery_drawdown_pct

    def reset(self, equity: float) -> None:
        """
        Manually reset the breaker to NORMAL state.

        Args:
            equity: The equity level to set as the new peak.
        """
        self.state = BreakerState.NORMAL
        self.peak_equity = equity
        self.current_equity = equity
        self.halted_at = None
        self.halt_reasons.clear()
        self._breach_count = 0
        # Apply cooldown after reset
        if self.cooldown_seconds > 0:
            self._cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=self.cooldown_seconds)
        else:
            self._cooldown_until = datetime.min.replace(tzinfo=timezone.utc)
        logger.info(
            "Circuit breaker RESET",
            name=self.name,
            equity=equity,
            cooldown_until=self._cooldown_until,
        )

    @property
    def is_halted(self) -> bool:
        """Indicates whether the breaker is currently halted."""
        return self.state == BreakerState.HALTED

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction of peak equity."""
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)