"""Drawdown-based circuit breakers — halt trading at configurable thresholds."""
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

    def update(self, equity: float) -> bool:
        """Call on every equity snapshot. Returns True if still NORMAL."""
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.current_equity = equity

        if self.state == BreakerState.HALTED:
            return False

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown >= self.max_drawdown_pct:
                self.state = BreakerState.HALTED
                self.halted_at = datetime.now(timezone.utc)
                reason = f"Drawdown {drawdown:.2%} >= threshold {self.max_drawdown_pct:.2%}"
                self.halt_reasons.append(reason)
                logger.error("Circuit breaker TRIPPED", name=self.name, drawdown=drawdown, threshold=self.max_drawdown_pct)
                return False

        return True

    def reset(self, equity: float) -> None:
        self.state = BreakerState.NORMAL
        self.peak_equity = equity
        self.halted_at = None
        logger.info("Circuit breaker RESET", name=self.name)

    @property
    def is_halted(self) -> bool:
        return self.state == BreakerState.HALTED

    @property
    def current_drawdown(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)
