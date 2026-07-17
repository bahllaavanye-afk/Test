"""Drawdown-based circuit breakers — halt trading at configurable thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    It can also automatically recover when the drawdown falls below a recovery threshold
    for a configurable number of consecutive checks (recovery_confirmation_period).
    An optional drawdown_rate_threshold adds a filter on the speed of equity decline.
    """
    name: str
    max_drawdown_pct: float                     # e.g. 0.10 = 10%
    peak_equity: float = 0.0
    current_equity: float = 0.0
    state: BreakerState = BreakerState.NORMAL
    halted_at: Optional[datetime] = None
    halt_reasons: List[str] = field(default_factory=list)

    # Configurable parameters
    confirmation_period: int = 1                # consecutive breaches required to halt
    recovery_drawdown_pct: float = 0.0          # drawdown pct below which the breaker may reset
    recovery_confirmation_period: int = 1      # consecutive checks below recovery threshold
    drawdown_rate_threshold: Optional[float] = None  # minimum drawdown change per update (fraction)

    # Internal tracking (not part of the public dataclass fields)
    _breach_count: int = field(init=False, default=0)
    _recovery_counter: int = field(init=False, default=0)
    _prev_drawdown: float = field(init=False, default=0.0)

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

        # Update peak and current equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.current_equity = equity

        # Compute current drawdown
        drawdown = self.current_drawdown

        # If already halted, evaluate recovery logic
        if self.state == BreakerState.HALTED:
            if self._should_recover(drawdown):
                self.reset(equity)
                logger.info("Circuit breaker auto‑recovered", name=self.name)
                return True
            return False

        # Determine drawdown rate (change since last update)
        drawdown_rate = drawdown - self._prev_drawdown
        self._prev_drawdown = drawdown

        # Check entry conditions
        breach = drawdown >= self.max_drawdown_pct
        rate_ok = (
            self.drawdown_rate_threshold is None
            or drawdown_rate >= self.drawdown_rate_threshold
        )
        if breach and rate_ok:
            self._breach_count += 1
            logger.debug(
                "Circuit breaker drawdown breach",
                name=self.name,
                drawdown=drawdown,
                threshold=self.max_drawdown_pct,
                breach_count=self._breach_count,
                drawdown_rate=drawdown_rate,
            )
            if self._breach_count >= self.confirmation_period:
                self.state = BreakerState.HALTED
                self.halted_at = datetime.now(timezone.utc)
                reason = (
                    f"Drawdown {drawdown:.2%} >= threshold {self.max_drawdown_pct:.2%} "
                    f"(confirmed {self._breach_count}×, rate {drawdown_rate:.2%})"
                )
                self.halt_reasons.append(reason)
                logger.error(
                    "Circuit breaker TRIPPED",
                    name=self.name,
                    drawdown=drawdown,
                    threshold=self.max_drawdown_pct,
                )
                return False
        else:
            if self._breach_count:
                logger.debug(
                    "Circuit breaker breach counter reset",
                    name=self.name,
                    previous_breach_count=self._breach_count,
                )
            self._breach_count = 0

        # Reset recovery counter on successful normal operation
        self._recovery_counter = 0
        return True

    def _should_recover(self, drawdown: float) -> bool:
        """
        Determine whether the breaker should automatically recover.

        Recovery occurs when the current drawdown falls below `recovery_drawdown_pct`
        for `recovery_confirmation_period` consecutive checks.
        """
        if self.recovery_drawdown_pct <= 0.0:
            return False
        if drawdown <= self.recovery_drawdown_pct:
            self._recovery_counter += 1
            logger.debug(
                "Circuit breaker recovery counter increment",
                name=self.name,
                drawdown=drawdown,
                recovery_threshold=self.recovery_drawdown_pct,
                recovery_counter=self._recovery_counter,
            )
            return self._recovery_counter >= self.recovery_confirmation_period
        else:
            if self._recovery_counter:
                logger.debug(
                    "Circuit breaker recovery counter reset",
                    name=self.name,
                    previous_counter=self._recovery_counter,
                )
            self._recovery_counter = 0
            return False

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
        self._recovery_counter = 0
        self._prev_drawdown = 0.0
        logger.info("Circuit breaker RESET", name=self.name, equity=equity)

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