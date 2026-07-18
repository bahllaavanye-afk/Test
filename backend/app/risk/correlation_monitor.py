"""
Cross‑strategy correlation monitor.

Hedge‑fund standard: if two strategies have rolling 5‑day return correlation
greater than 0.70, automatically reduce the smaller one (by total PnL) by 50%
until the correlation drops below 0.50. This prevents correlated drawdowns –
the #1 unaddressed risk in multi‑strategy bots.
"""

from __future__ import annotations

import asyncio
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

import numpy as np

from app.utils.logging import logger


@dataclass
class CorrelationAlert:
    """
    Data container for a correlation‑based risk alert.

    Attributes
    ----------
    strategy_a: str
        Name of the first strategy in the correlated pair.
    strategy_b: str
        Name of the second strategy in the correlated pair.
    correlation: float
        The computed correlation coefficient between the two strategies.
    action: str
        The action taken – ``reduce_b``, ``reduce_a`` or ``monitor``.
    reduced_strategy: Optional[str]
        The strategy that was halved, if any.
    timestamp: datetime
        UTC timestamp when the alert was generated.
    """

    strategy_a: str
    strategy_b: str
    correlation: float
    action: str  # 'reduce_b' | 'reduce_a' | 'monitor'
    reduced_strategy: Optional[str]
    timestamp: datetime

    def to_dict(self) -> dict:
        """Return a JSON‑serialisable representation of the alert."""
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
    Monitor that tracks rolling returns of multiple strategies and raises alerts
    when pairwise correlations exceed predefined thresholds.

    The monitor maintains a fixed‑size rolling window of recent returns for each
    strategy, periodically computes a correlation matrix, and halves the exposure
    of the smaller strategy when the correlation is too high. When the correlation
    falls back below the resume threshold, the reduction is lifted.
    """

    def __init__(
        self,
        window: int = 5,
        kill_threshold: float = 0.70,
        resume_threshold: float = 0.50,
        scan_interval: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        window: int, default 5
            Number of most recent returns to keep for each strategy.
        kill_threshold: float, default 0.70
            Correlation value above which a reduction is triggered.
        resume_threshold: float, default 0.50
            Correlation value below which a previous reduction is lifted.
        scan_interval: int, default 60
            Seconds between successive correlation scans.
        """
        self.window = window
        self.kill_threshold = kill_threshold
        self.resume_threshold = resume_threshold
        self.scan_interval = scan_interval
        self._returns: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._reduced: set[str] = set()  # strategies currently halved
        self._alerts: deque[CorrelationAlert] = deque(maxlen=200)
        self._running = False

    def record_return(self, strategy: str, ret: float) -> None:
        """
        Append a new return observation for a given strategy.

        Parameters
        ----------
        strategy: str
            Identifier of the strategy.
        ret: float
            Return value for the latest bar/period.
        """
        self._returns[strategy].append(ret)

    def correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise Pearson correlation coefficients for all strategy pairs
        that have at least three recorded returns.

        Returns
        -------
        Dict[Tuple[str, str], float]
            Mapping from strategy pair to correlation coefficient.
        """
        strategies = [s for s, r in self._returns.items() if len(r) >= 3]
        result: Dict[Tuple[str, str], float] = {}
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

    def scan(self) -> List[CorrelationAlert]:
        """
        Perform a single scan of the correlation matrix and generate alerts.

        Returns
        -------
        List[CorrelationAlert]
            Alerts generated during this scan (may be empty).
        """
        matrix = self.correlation_matrix()
        new_alerts: List[CorrelationAlert] = []
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
                        logger.info(f"CORR RESUME: {s} correlation normalized (corr={corr:.2f})")
        return new_alerts

    def is_reduced(self, strategy: str) -> bool:
        """
        Check whether a strategy is currently operating at a reduced size.

        Parameters
        ----------
        strategy: str
            Identifier of the strategy.

        Returns
        -------
        bool
            ``True`` if the strategy is halved, otherwise ``False``.
        """
        return strategy in self._reduced

    def sizing_multiplier(self, strategy: str) -> float:
        """
        Retrieve the sizing multiplier for a strategy based on reduction state.

        Parameters
        ----------
        strategy: str
            Identifier of the strategy.

        Returns
        -------
        float
            ``0.5`` if the strategy is reduced, else ``1.0``.
        """
        return 0.5 if strategy in self._reduced else 1.0

    def recent_alerts(self, limit: int = 20) -> List[dict]:
        """
        Get the most recent alerts as dictionaries.

        Parameters
        ----------
        limit: int, default 20
            Maximum number of alerts to return.

        Returns
        -------
        List[dict]
            Alert dictionaries ordered from oldest to newest within the limit.
        """
        return [a.to_dict() for a in list(self._alerts)[-limit:]]

    def matrix_as_list(self) -> List[dict]:
        """
        Represent the current correlation matrix as a list of dictionaries.

        Returns
        -------
        List[dict]
            Each entry contains ``strategy_a``, ``strategy_b`` and rounded
            ``correlation``.
        """
        return [
            {"strategy_a": k[0], "strategy_b": k[1], "correlation": round(v, 3)}
            for k, v in self.correlation_matrix().items()
        ]

    async def run_forever(self) -> None:
        """
        Continuously execute scans at the configured interval until stopped.

        This coroutine runs indefinitely; call :meth:`stop` to terminate it.
        """
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
            except Exception as e:
                logger.error(f"CorrelationMonitor scan error: {e}")
            await asyncio.sleep(self.scan_interval)

    def stop(self) -> None:
        """
        Signal the monitor to stop its asynchronous run loop.
        """
        self._running = False


correlation_monitor = CrossStrategyCorrelationMonitor()