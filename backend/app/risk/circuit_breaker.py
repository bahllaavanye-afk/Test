"""Drawdown-based circuit breakers — halt trading at configurable thresholds.

The circuit breaker monitors equity drawdown and can halt trading when the drawdown
exceeds a configured maximum percentage.  A confirmation period can be set to
require multiple consecutive breaches before the breaker trips, reducing the
likelihood of false positives.  An optional recovery threshold allows the breaker
to automatically reset once drawdown improves.

Typical usage::

    breaker = CircuitBreaker(name="EquityCB", max_drawdown_pct=0.10,
                             confirmation_period=3, recovery_drawdown_pct=0.05)
    if not breaker.update(current_equity):
        # trading is halted
        ...

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from app.utils.logging import logger


class BreakerState(str, Enum):
    """Possible states of a :class:`CircuitBreaker`."""

    NORMAL = "normal"
    """The breaker is not tripped; trading may continue."""

    HALTED = "halted"
    """The breaker has tripped and trading should be halted."""


@dataclass
class CircuitBreaker:
    """
    Monitor equity drawdown and halt trading when configured thresholds are breached.

    Attributes
    ----------
    name: str
        Identifier for the breaker (used in log messages).
    max_drawdown_pct: float
        Maximum allowed drawdown expressed as a fraction (e.g. ``0.10`` for 10 %).
    peak_equity: float
        Highest equity observed since the last reset.  Updated automatically.
    current_equity: float
        Most recent equity value supplied to :meth:`update`.
    state: BreakerState
        Current state of the breaker – either ``NORMAL`` or ``HALTED``.
    halted_at: Optional[datetime]
        Timestamp when the breaker entered the ``HALTED`` state; ``None`` if not halted.
    halt_reasons: List[str]
        Human‑readable reasons accumulated each time the breaker trips.
    confirmation_period: int
        Number of consecutive drawdown breaches required before the breaker trips.
    recovery_drawdown_pct: float
        Drawdown fraction below which the breaker automatically recovers.
    _breach_count: int
        Internal counter of consecutive breaches (not part of the public API).
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
        Process a new equity snapshot and update the breaker state.

        Parameters
        ----------
        equity: float
            Latest equity value.  Must be a numeric type; otherwise the update is ignored.

        Returns
        -------
        bool
            ``True`` if the breaker remains in the ``NORMAL`` state,
            ``False`` if it is currently ``HALTED``.
        """
        if equity is None or not isinstance(equity, (int, float)):
            logger.warning("Circuit breaker received invalid equity value", name=self.name, equity=equity)
            return not self.is_halted

        # Update peak and current equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.current_equity = equity

        # If already halted, check for possible auto‑recovery
        if self.state == BreakerState.HALTED:
            if self._should_recover():
                self.reset(equity)
                logger.info("Circuit breaker auto‑recovered", name=self.name)
            else:
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

        Recovery occurs when the current drawdown falls below ``recovery_drawdown_pct``.

        Returns
        -------
        bool
            ``True`` if recovery conditions are met, otherwise ``False``.
        """
        if self.recovery_drawdown_pct <= 0.0:
            return False
        return self.current_drawdown <= self.recovery_drawdown_pct

    def reset(self, equity: float) -> None:
        """
        Manually reset the breaker to the ``NORMAL`` state.

        Parameters
        ----------
        equity: float
            The equity level to set as the new peak after reset.
        """
        self.state = BreakerState.NORMAL
        self.peak_equity = equity
        self.current_equity = equity
        self.halted_at = None
        self.halt_reasons.clear()
        self._breach_count = 0
        logger.info("Circuit breaker RESET", name=self.name, equity=equity)

    @property
    def is_halted(self) -> bool:
        """bool: ``True`` if the breaker is currently in the ``HALTED`` state."""
        return self.state == BreakerState.HALTED

    @property
    def current_drawdown(self) -> float:
        """float: Current drawdown as a fraction of peak equity (0.0 if no peak)."""
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)