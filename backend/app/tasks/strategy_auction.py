"""
Strategy Auction — UCB1 Bandit Capital Allocation.

Strategies compete for capital allocation based on proven performance.
Better-performing strategies automatically receive more capital.
Underperformers are defunded until they can demonstrate improvement.

Algorithm:
  UCB1 score = avg_sharpe + sqrt(2 * ln(total_pulls) / strategy_pulls)
  - avg_sharpe = exploitation term (proven performance)
  - sqrt(...) = exploration bonus (ensures every strategy gets tried)

Capital is re-allocated every hour from the auction:
  - Top 30% of strategies by UCB1 score share 70% of risk budget
  - Bottom 70% share remaining 30%
  - Strategies with zero Sharpe for 7+ days enter "probation" (1% allocation)

All auction results are published to agent_bus topic "auction:allocated"
so risk engine, strategy runner, and knowledge loop all react in real time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import heapq
from typing import Any

logger = logging.getLogger(__name__)

_AUCTION_STATE_KEY = "auction:state"
_ALLOCATION_KEY = "auction:allocations"
_MIN_PULLS_FOR_EXPLOIT = 5      # need at least 5 runs before exploiting Sharpe
_PROBATION_THRESHOLD_DAYS = 7   # days of zero/negative Sharpe before probation
_PROBATION_ALLOCATION = 0.01    # 1% allocation for probation strategies


class StrategyBid:
    """Tracks a strategy's performance history for UCB1 scoring."""

    def __init__(self, strategy_name: str) -> None:
        self.name = strategy_name
        self.pulls: int = 0             # how many times it has run
        self.total_sharpe: float = 0.0  # sum of all Sharpe observations
        self.last_sharpe: float = 0.0
        self.consecutive_bad_days: int = 0
        self.first_seen: float = time.time()
        self.last_run: float = 0.0

    @property
    def avg_sharpe(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_sharpe / self.pulls

    def ucb1_score(self, total_pulls: int) -> float:
        if self.pulls < _MIN_PULLS_FOR_EXPLOIT:
            # Exploration bonus is very high for new strategies — try them first
            return 10.0 + (1.0 / (self.pulls + 1))
        if total_pulls == 0 or self.pulls == 0:
            return 0.0
        exploration = math.sqrt(2 * math.log(total_pulls) / self.pulls)
        return self.avg_sharpe + exploration

    def is_on_probation(self) -> bool:
        return self.pulls >= _MIN_PULLS_FOR_EXPLOIT and self.consecutive_bad_days >= _PROBATION_THRESHOLD_DAYS

    def record_run(self, sharpe: float) -> None:
        self.pulls += 1
        self.total_sharpe += sharpe
        self.last_sharpe = sharpe
        self.last_run = time.time()
        if sharpe <= 0:
            self.consecutive_bad_days += 1
        else:
            self.consecutive_bad_days = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pulls": self.pulls,
            "total_sharpe": self.total_sharpe,
            "last_sharpe": self.last_sharpe,
            "consecutive_bad_days": self.consecutive_bad_days,
            "first_seen": self.first_seen,
            "last_run": self.last_run,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyBid":
        bid = cls(d["name"])
        bid.pulls = int(d.get("pulls", 0))
        bid.total_sharpe = float(d.get("total_sharpe", 0.0))
        bid.last_sharpe = float(d.get("last_sharpe", 0.0))
        bid.consecutive_bad_days = int(d.get("consecutive_bad_days", 0))
        bid.first_seen = float(d.get("first_seen", time.time()))
        bid.last_run = float(d.get("last_run", 0.0))
        return bid


class StrategyAuction:
    """
    Runs the capital allocation auction every hour.

    Reads strategy performance from AgentMemory, computes UCB1 scores,
    and publishes allocation decisions to the agent bus.
    """

    def __init__(self, redis_client: Any, total_capital_usd: float = 10_000.0) -> None:
        self._r = redis_client
        self._total_capital = total_capital_usd
        self._bids: dict[str, StrategyBid] = {}
        self._loaded = False

    # ── Persistence ───────────────────────────────────────────────────────────

    async def _load_state(self) -> None:
        try:
            raw = await self._r.get(_AUCTION_STATE_KEY)
            if raw:
                data = json.loads(raw)
                self._bids = {name: StrategyBid.from_dict(d) for name, d in data.items()}
                logger.info("StrategyAuction: loaded %d strategy bids", len(self._bids))
        except Exception as e:
            logger.debug("StrategyAuction._load_state: %s", e)
        self._loaded = True

    async def _save_state(self) -> None:
        try:
            data = {name: bid.to_dict() for name, bid in self._bids.items()}
            await self._r.set(_AUCTION_STATE_KEY, json.dumps(data), ex=86400 * 30)
        except Exception as e:
            logger.debug("StrategyAuction._save_state: %s", e)

    # ── Recording performance ─────────────────────────────────────────────────

    async def record_performance(self, strategy_name: str, sharpe: float) -> None:
        """Called by strategy_runner after each evaluation cycle."""
        if not self._loaded:
            await self._load_state()
        if strategy_name not in self._bids:
            self._bids[strategy_name] = StrategyBid(strategy_name)
        self._bids[strategy_name].record_run(sharpe)
        await self._save_state()

    # ── Running the auction ───────────────────────────────────────────────────

    async def run_auction(self) -> dict[str, float]:
        """
        Compute UCB1 allocations for all known strategies.
        Returns: {strategy_name: capital_usd}
        Publishes auction:allocated to agent bus.
        """
        if not self._loaded:
            await self._load_state()

        if not self._bids:
            logger.info("StrategyAuction: no strategies registered yet")
            return {}

        total_pulls = sum(b.pulls for b in self._bids.values())

        # Separate probation and non‑probation strategies while computing scores
        non_probation: list[tuple[str, float]] = []
        probation: list[tuple[str, float]] = []

        for name, bid in self._bids.items():
            score = bid.ucb1_score(total_pulls)
            if bid.is_on_probation():
                probation.append((name, score))
            else:
                non_probation.append((name, score))

        # Determine top 30% of non‑probation strategies efficiently
        top_count = max(1, len(non_probation) * 3 // 10)
        if top_count < len(non_probation):
            top_strategies = heapq.nlargest(top_count, non_probation, key=lambda x: x[1])
            # Remaining strategies are those not in the top set
            top_names = {n for n, _ in top_strategies}
            rest_strategies = [(n, s) for n, s in non_probation if n not in top_names]
        else:
            top_strategies = non_probation
            rest_strategies = []

        probation_capital = len(probation) * self._total_capital * _PROBATION_ALLOCATION
        available_capital = max(0, self._total_capital - probation_capital)

        top_capital = available_capital * 0.70
        rest_capital = available_capital * 0.30

        def distribute(strategies: list[tuple[str, float]], budget: float) -> dict[str, float]:
            if not strategies:
                return {}
            # Ensure a minimal positive weight to avoid division by zero
            total_score = sum(max(s, 0.001) for _, s in strategies)
            return {n: budget * max(s, 0.001) / total_score for n, s in strategies}

        allocations: dict[str, float] = {}
        allocations.update(distribute(top_strategies, top_capital))
        allocations.update(distribute(rest_strategies, rest_capital))
        for name, _ in probation:
            allocations[name] = self._total_capital * _PROBATION_ALLOCATION

        # Persist allocations for strategy_runner to read
        try:
            await self._r.set(_ALLOCATION_KEY, json.dumps(allocations), ex=7200)
        except Exception as e:
            logger.debug("StrategyAuction: failed to save allocations: %s", e)

        return allocations

# ... (truncated for brevity)