"""
Cross-strategy correlation monitor.
Hedge fund standard: if two strategies have rolling 5-day return correlation > 0.70,
auto-reduce the smaller one (by total_pnl) by 50% until correlation drops below 0.50.

This prevents correlated drawdowns — the #1 unaddressed risk in multi-strategy bots.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
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
        self.window = window
        self.kill_threshold = kill_threshold
        self.resume_threshold = resume_threshold
        self.scan_interval = scan_interval
        self._returns: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._reduced: set[str] = set()  # strategies currently halved
        self._alerts: deque[CorrelationAlert] = deque(maxlen=200)
        self._running = False

    def record_return(self, strategy: str, ret: float) -> None:
        """Append a new return value for a given strategy."""
        self._returns[strategy].append(ret)

    def correlation_matrix(self) -> dict[tuple[str, str], float]:
        """Compute pairwise correlations for all strategies with enough data."""
        strategies = [s for s, r in self._returns.items() if len(r) >= 3]
        result: dict[tuple[str, str], float] = {}
        for i, s_a in enumerate(strategies):
            for s_b in strategies[i + 1 :]:
                corr = self._pair_correlation(s_a, s_b)
                if corr is not None:
                    result[(s_a, s_b)] = corr
        return result

    def _pair_correlation(self, s_a: str, s_b: str) -> Optional[float]:
        """Return correlation for a pair of strategies or None if not computable."""
        r_a = list(self._returns[s_a])
        r_b = list(self._returns[s_b])
        min_len = min(len(r_a), len(r_b))
        if min_len < 3:
            return None
        r_a, r_b = r_a[-min_len:], r_b[-min_len:]
        if np.std(r_a) == 0 or np.std(r_b) == 0:
            return None
        return float(np.corrcoef(r_a, r_b)[0, 1])

    def scan(self) -> list[CorrelationAlert]:
        """Perform a single correlation scan and emit alerts if thresholds are crossed."""
        alerts: list[CorrelationAlert] = []
        matrix = self.correlation_matrix()
        for (s_a, s_b), corr in matrix.items():
            if corr > self.kill_threshold:
                alerts.extend(self._handle_kill(s_a, s_b, corr))
            elif corr < self.resume_threshold:
                self._handle_resume(s_a, s_b, corr)
        return alerts

    def _handle_kill(self, s_a: str, s_b: str, corr: float) -> list[CorrelationAlert]:
        """
        Reduce the smaller strategy when correlation exceeds kill_threshold.
        Returns any generated alerts.
        """
        # Choose the strategy with fewer recorded returns as the smaller one.
        smaller = s_b if len(self._returns[s_a]) >= len(self._returns[s_b]) else s_a
        if smaller in self._reduced:
            return []
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
        logger.warning(
            f"CORR KILL-SWITCH: {s_a}↔{s_b} corr={corr:.2f} > {self.kill_threshold}. Halving {smaller}."
        )
        return [alert]

    def _handle_resume(self, s_a: str, s_b: str, corr: float) -> None:
        """
        Re‑enable strategies whose correlation fell below resume_threshold.
        """
        for s in (s_a, s_b):
            if s in self._reduced:
                self._reduced.discard(s)
                logger.info(f"CORR RESUME: {s} correlation normalized (corr={corr:.2f})")

    def is_reduced(self, strategy: str) -> bool:
        """Return True if the strategy is currently halved."""
        return strategy in self._reduced

    def sizing_multiplier(self, strategy: str) -> float:
        """Return the sizing multiplier for a strategy (0.5 if reduced, else 1.0)."""
        return 0.5 if strategy in self._reduced else 1.0

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        """Return the most recent alerts as dictionaries."""
        return [a.to_dict() for a in list(self._alerts)[-limit:]]

    def matrix_as_list(self) -> list[dict]:
        """Return the correlation matrix as a list of dictionaries for UI consumption."""
        return [
            {"strategy_a": k[0], "strategy_b": k[1], "correlation": round(v, 3)}
            for k, v in self.correlation_matrix().items()
        ]

    async def run_forever(self) -> None:
        """Continuously scan at the configured interval until stopped."""
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
                            f"Halved {a.reduced_strategy}: corr {a.correlation:.2f} with {a.strategy_a}↔{a.strategy_b}",
                        )
            except Exception as e:  # pragma: no cover
                logger.error(f"CorrelationMonitor scan error: {e}")
            await asyncio.sleep(self.scan_interval)

    def stop(self) -> None:
        """Signal the monitoring loop to stop."""
        self._running = False


correlation_monitor = CrossStrategyCorrelationMonitor()