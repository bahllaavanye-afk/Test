"""Drawdown-based circuit breakers — halt trading at configurable thresholds.

This module defines a simple circuit breaker that monitors equity drawdown and
halts trading when the drawdown exceeds a configured threshold. It provides
methods to update the equity snapshot, reset the breaker, and query its state.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from numbers import Number

from app.utils.logging import logger


class BreakerState(str, Enum):
    """Enumeration of possible circuit‑breaker states.

    Attributes
    ----------
    NORMAL : str
        The breaker is active and trading may continue.
    HALTED : str
        The breaker has tripped; trading should be halted.
    """

    NORMAL = "normal"
    HALTED = "halted"


@dataclass
class CircuitBreaker:
    """Circuit breaker that halts trading based on equity drawdown.

    Parameters
    ----------
    name : str
        Identifier for the circuit breaker instance.
    max_drawdown_pct : float
        Maximum allowed drawdown as a fraction (e.g., 0.10 for 10%).
    peak_equity : float, optional
        Highest equity observed; initialised to ``0.0``.
    current_equity : float, optional
        Most recent equity snapshot; initialised to ``0.0``.
    state : BreakerState, optional
        Current breaker state; defaults to :class:`BreakerState.NORMAL`.
    halted_at : datetime | None, optional
        Timestamp when the breaker last tripped; ``None`` if not halted.
    halt_reasons : list[str], optional
        Accumulated reasons for halting; starts empty.
    """

    name: str
    max_drawdown_pct: float          # e.g. 0.10 = 10%
    peak_equity: float = 0.0
    current_equity: float = 0.0
    state: BreakerState = BreakerState.NORMAL
    halted_at: datetime | None = None
    halt_reasons: list[str] = field(default_factory=list)

    def _validate_equity(self, equity: float) -> None:
        """Validate that ``equity`` is a numeric, non‑negative value.

        Raises
        ------
        TypeError
            If ``equity`` is not a numeric type.
        ValueError
            If ``equity`` is negative.
        """
        if not isinstance(equity, Number):
            raise TypeError(f"Equity must be a numeric type, got {type(equity).__name__}")
        if equity < 0:
            raise ValueError("Equity cannot be negative")

    def update(self, equity: float) -> bool:
        """Process a new equity snapshot.

        Updates internal peak and current equity values and checks whether the
        drawdown exceeds ``max_drawdown_pct``. If the breaker trips, it records
        the halt time and reason.

        Parameters
        ----------
        equity : float
            Latest equity value.

        Returns
        -------
        bool
            ``True`` if the breaker remains in the ``NORMAL`` state; ``False`` if
            it is ``HALTED`` or an error occurred.
        """
        try:
            self._validate_equity(equity)

            if equity > self.peak_equity:
                self.peak_equity = equity
            self.current_equity = equity

            if self.state == BreakerState.HALTED:
                return False

            if self.peak_equity > 0:
                drawdown = (self.peak_equity - equity) / self.peak_equity
                if drawdown >= self.max_drawdown_pct:
                    self.state = BreakerState.HALTED
                    self.halted_at = datetime.now(UTC)
                    reason = f"Drawdown {drawdown:.2%} >= threshold {self.max_drawdown_pct:.2%}"
                    self.halt_reasons.append(reason)
                    logger.error(
                        "Circuit breaker TRIPPED",
                        name=self.name,
                        drawdown=drawdown,
                        threshold=self.max_drawdown_pct,
                        reason=reason,
                    )
                    return False

            return True
        except (TypeError, ValueError) as e:
            logger.error(
                "Circuit breaker update validation error",
                name=self.name,
                error=str(e),
                equity=equity,
            )
            return False
        except Exception as e:
            logger.exception(
                "Unexpected error during circuit breaker update",
                name=self.name,
                equity=equity,
                error=str(e),
            )
            return False

    def reset(self, equity: float) -> None:
        """Reset the circuit breaker to a normal state using the provided equity.

        Parameters
        ----------
        equity : float
            Equity value to initialise ``peak_equity`` after reset.
        """
        try:
            self._validate_equity(equity)
            self.state = BreakerState.NORMAL
            self.peak_equity = equity
            self.halted_at = None
            self.halt_reasons = []
            logger.info("Circuit breaker RESET", name=self.name, equity=equity)
        except (TypeError, ValueError) as e:
            logger.error(
                "Circuit breaker reset validation error",
                name=self.name,
                error=str(e),
                equity=equity,
            )
        except Exception as e:
            logger.exception(
                "Unexpected error during circuit breaker reset",
                name=self.name,
                equity=equity,
                error=str(e),
            )

    @property
    def is_halted(self) -> bool:
        """Indicates whether the breaker is currently halted."""
        return self.state == BreakerState.HALTED

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction of ``peak_equity``.

        Returns ``0.0`` if ``peak_equity`` is zero.
        """
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)