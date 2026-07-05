"""
Cross-strategy correlation monitor.
Hedge fund standard: if two strategies have rolling 5-day return correlation > 0.70,
auto-reduce the smaller one (by total_pnl) by 50% until correlation drops below 0.50.

This prevents correlated drawdowns — the #1 unaddressed risk in multi-strategy bots.
"""
from __future__ import annotations

import asyncio
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Optional

import numpy as np

from app.utils.logging import logger


@dataclass
class CorrelationAlert:
    strategy_a: str
    strategy_b: str
    correlation: float
    action: str  # 'reduce_b' | 'reduce_a' | 'monitor'
    reduced_strategy: Optional[str]
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "correlation": round(self.correlation, 3),
            "action": self.action,
            "reduced_strategy": self.reduced_strategy,
            "timestamp": self.timestamp.isoformat(),
        }


class CrossStrategyCorrelationMonitor:
    """
    Tracks per-strategy returns in a rolling window.
    Runs correlation scan every N seconds.
    When correlation > kill_threshold, fires a CorrelationAlert.
    """

    def __init__(
        self,
        window: int = 5,  # rolling 5 bars
        kill_threshold: float = 0.70,
        resume_threshold: float = 0.50,
        scan_interval: int = 60,
    ):
        # ---- validation ----
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")
        if not isinstance(kill_threshold, Real) or not (0 <= kill_threshold <= 1):
            raise ValueError("kill_threshold must be a float in the range [0, 1]")
        if not isinstance(resume_threshold, Real) or not (0 <= resume_threshold <= 1):
            raise ValueError("resume_threshold must be a float in the range [0, 1]")
        if resume_threshold >= kill_threshold:
            raise ValueError("resume_threshold must be less than kill_threshold")
        if not isinstance(scan_interval, int) or scan_interval <= 0:
            raise ValueError("scan_interval must be a positive integer")

        self.window = window
        self.kill_threshold = kill_threshold
        self.resume_threshold = resume_threshold
        self.scan_interval = scan_interval
        self._returns: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._reduced: set[str] = set()  # strategies currently halved
        self._alerts: deque[CorrelationAlert] = deque(maxlen=200)
        self._running = False

    def record_return(self, strategy: str, ret: float) -> None:
        """Record a single return value for a strategy."""
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("strategy must be a non‑empty string")
        if not isinstance(ret, Real):
            raise ValueError("ret must be a numeric type")
        self._returns[strategy].append(float(ret))

    def correlation_matrix(self) -> dict[tuple[str, str], float]:
        """Compute pairwise correlations for strategies with enough data."""
        strategies = [s for s, r in self._returns.items() if len(r) >= 3]
        result: dict[tuple[str, str], float] = {}
        for i, s_a in enumerate(strategies):
            for s_b in strategies[i + 1 :]:
                r_a = list(self._returns[s_a])
                r_b = list(self._returns[s_b])
                min_len = min(len(r_a), len(r_b))
                if min_len < 3:
                    continue
                r_a, r_b = r_a[-min_len:], r_b[-min_len:]
                if np.std(r_a) == 0 or np.std(r_b) == 0:
                    continue
                corr = float(np.corrcoef(r_a, r_b)[0, 1])
                result[(s_a, s_b)] = corr
        return result

    def scan(self) -> list[CorrelationAlert]:
        """Scan the correlation matrix and generate alerts if thresholds are crossed."""
        matrix = self.correlation_matrix()
        new_alerts: list[CorrelationAlert] = []
        for (s_a, s_b), corr in matrix.items():
            if corr > self.kill_threshold:
                # Reduce the strategy with fewer returns recorded (proxy for smaller)
                smaller = s_b if len(self._returns[s_a]) >= len(self._returns[s_b]) else s_a
                if smaller not in self._reduced:
                    self._reduced.add(smaller)
                    alert = CorrelationAlert(
                        strategy_a=s_a,
                        strategy_b=s_b,
                        correlation=corr,
                        action=f"reduce_{smaller.split('_')[0]}",
                        reduced_strategy=smaller,
                        timestamp=datetime.now(timezone.utc),
                    )
                    self._alerts.append(alert)
                    new_alerts.append(alert)
                    logger.warning(
                        f"CORR KILL-SWITCH: {s_a}↔{s_b} corr={corr:.2f} > {self.kill_threshold}. "
                        f"Halving {smaller}."
                    )
            elif corr < self.resume_threshold:
                # Re‑enable if correlation dropped
                for s in (s_a, s_b):
                    if s in self._reduced:
                        self._reduced.discard(s)
                        logger.info(
                            f"CORR RESUME: {s} correlation normalized (corr={corr:.2f})"
                        )
        return new_alerts

    def is_reduced(self, strategy: str) -> bool:
        """Return True if the strategy is currently reduced."""
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("strategy must be a non‑empty string")
        return strategy in self._reduced

    def sizing_multiplier(self, strategy: str) -> float:
        """Return the sizing multiplier for a strategy (0.5 if reduced, else 1.0)."""
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("strategy must be a non‑empty string")
        return 0.5 if strategy in self._reduced else 1.0

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        """Return the most recent alerts as dictionaries, limited by `limit`."""
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return [a.to_dict() for a in list(self._alerts)[-limit:]]

    def matrix_as_list(self) -> list[dict]:
        """Return the correlation matrix as a list of dictionaries for UI consumption."""
        return [
            {"strategy_a": k[0], "strategy_b": k[1], "correlation": round(v, 3)}
            for k, v in self.correlation_matrix().items()
        ]

    async def run_forever(self) -> None:
        """Continuously run scans at the configured interval until stopped."""
        self._running = True
        while self._running:
            try:
                alerts = self.scan()
                if alerts:
                    from app.notifications.tracker import tracker

                    for a in alerts:
                        tracker.record(
                            "correlation_kill_switch",
                            "risk",
                            f"Halved {a.reduced_strategy}: corr {a.correlation:.2f} "
                            f"with {a.strategy_a}↔{a.strategy_b}",
                        )
            except Exception as e:  # pragma: no cover
                logger.error(f"CorrelationMonitor scan error: {e}")
            await asyncio.sleep(self.scan_interval)

    def stop(self) -> None:
        """Stop the background run loop."""
        self._running = False


correlation_monitor = CrossStrategyCorrelationMonitor()