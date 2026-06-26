"""Drawdown-based circuit breakers — halt trading at configurable thresholds."""
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from numbers import Number

from app.utils.logging import logger


class BreakerState(str, Enum):
    NORMAL = "normal"
    HALTED = "halted"


@dataclass
class CircuitBreaker:
    name: str
    max_drawdown_pct: float          # e.g. 0.10 = 10%
    peak_equity: float = 0.0
    current_equity: float = 0.0
    state: BreakerState = BreakerState.NORMAL
    halted_at: datetime | None = None
    halt_reasons: list[str] = field(default_factory=list)

    def _validate_equity(self, equity: float) -> None:
        """Validate that equity is a numeric, non‑negative value."""
        if not isinstance(equity, Number):
            raise TypeError(f"Equity must be a numeric type, got {type(equity).__name__}")
        if equity < 0:
            raise ValueError("Equity cannot be negative")

    def update(self, equity: float) -> bool:
        """Call on every equity snapshot. Returns True if still NORMAL."""
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
        """Reset the circuit breaker to a normal state using the provided equity."""
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
        return self.state == BreakerState.HALTED

    @property
    def current_drawdown(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)