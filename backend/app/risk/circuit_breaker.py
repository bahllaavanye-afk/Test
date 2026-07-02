"""Drawdown-based circuit breakers — halt trading at configurable thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from app.utils.logging import logger


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""


class InvalidEquityError(CircuitBreakerError, TypeError):
    """Raised when an invalid equity value is provided."""


class RecoveryError(CircuitBreakerError, RuntimeError):
    """Raised when an unexpected error occurs during recovery evaluation."""


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
    """
    name: str
    max_drawdown_pct: float                     # e.g. 0.10 = 10%
    peak_equity: float = 0.0
    current_equity: float = 0.0
    state: BreakerState = BreakerState.NORMAL
    halted_at: Optional[datetime] = None
    halt_reasons: List[str] = field(default_factory=list)

    # New configurable parameters
    confirmation_period: int = 1                # number of consecutive breaches required to halt
    recovery_drawdown_pct: float = 0.0          # drawdown pct below which the breaker auto‑resets

    # Internal tracking (not part of the public dataclass fields)
    _breach_count: int = field(init=False, default=0)

    def update(self, equity: float) -> bool:
        """
        Update the breaker with the latest equity snapshot.

        Returns:
            bool: True if the breaker remains in NORMAL state, False if HALTED.
        """
        try:
            self._validate_equity(equity)
        except InvalidEquityError as exc:
            logger.error(
                "Circuit breaker received invalid equity value",
                name=self.name,
                equity=equity,
                error=str(exc),
                exc_info=True,
            )
            return not self.is_halted

        # Update peak and current equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.current_equity = equity

        # If already halted, check for possible auto‑recovery
        if self.state == BreakerState.HALTED:
            try:
                if self._should_recover():
                    self.reset(equity)
                    logger.info("Circuit breaker auto‑recovered", name=self.name)
                else:
                    return False
            except RecoveryError as exc:
                logger.error(
                    "Error during circuit breaker recovery evaluation",
                    name=self.name,
                    equity=equity,
                    error=str(exc),
                    exc_info=True,
                )
                return False

        # Compute drawdown only when peak_equity is positive
        drawdown = self.current_drawdown
        if drawdown >= self.max_drawdown_pct:
            self._breach_count += 1
            logger.debug(
                "Circuit breaker drawdown breach",
                name=self.name,
                drawdown=drawdown,
                threshold=self.max_drawdown_pct,
                breach_count=self._breach_count,
            )
            if self._breach_count >= self.confirmation_period:
                self.state = BreakerState.HALTED
                self.halted_at = datetime.now(timezone.utc)
                reason = (
                    f"Drawdown {drawdown:.2%} >= threshold {self.max_drawdown_pct:.2%} "
                    f"(confirmed {self._breach_count}×)"
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
            # Reset breach counter when drawdown falls back below threshold
            if self._breach_count:
                logger.debug(
                    "Circuit breaker breach counter reset",
                    name=self.name,
                    previous_breach_count=self._breach_count,
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
        try:
            return self.current_drawdown <= self.recovery_drawdown_pct
        except Exception as exc:
            raise RecoveryError("Failed to evaluate recovery condition") from exc

    def reset(self, equity: float) -> None:
        """
        Manually reset the breaker to NORMAL state.

        Args:
            equity: The equity level to set as the new peak.
        """
        try:
            self._validate_equity(equity)
        except InvalidEquityError as exc:
            logger.error(
                "Circuit breaker reset with invalid equity value",
                name=self.name,
                equity=equity,
                error=str(exc),
                exc_info=True,
            )
            raise

        self.state = BreakerState.NORMAL
        self.peak_equity = equity
        self.current_equity = equity
        self.halted_at = None
        self.halt_reasons.clear()
        self._breach_count = 0
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

    @staticmethod
    def _validate_equity(equity: float) -> None:
        """Validate equity input; raise InvalidEquityError if invalid."""
        if equity is None or not isinstance(equity, (int, float)):
            raise InvalidEquityError(f"Equity must be a numeric value, got {type(equity).__name__}")
        if equity < 0:
            raise InvalidEquityError("Equity cannot be negative")