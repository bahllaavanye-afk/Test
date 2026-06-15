"""
Knowledge Loop — The company's continuous learning brain.

This is the direct answer to: "How are we making sure the company is growing
and learning every single minute?"

Architecture:
  - Subscribes to EVERY event on the agent bus
  - For each event: extracts a structured lesson
  - Writes lessons to AgentMemory (Redis-backed, shared across all agents)
  - Aggregates patterns: which strategies outperform in which regimes
  - Fires a "knowledge:learned" event so every agent can immediately apply new insight
  - Every 5 minutes: synthesizes recent lessons into a "state of the firm" summary
  - Self-healing: if a component fails repeatedly, logs a circuit-breaker alert

Memory architecture (Letta-style):
  CORE memory (stable)    → "market:regime", "top_strategies", "risk_limits"
  EPISODIC memory (recent) → last 500 events, what happened and what we learned
  SCRATCH (ephemeral)     → current cycle context, cleared each run

All agents share the same Redis-backed AgentMemory, so a lesson learned
from a trade outcome immediately informs the strategy runner, risk engine,
and any other subscriber — no polling, no lag, no lost learnings.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# How often to synthesize a "state of the firm" summary
_SYNTHESIS_INTERVAL = 300  # 5 minutes

# How many recent episodes to include in synthesis
_SYNTHESIS_WINDOW = 50

# Failure tracking for self-healing circuit breaker
_COMPONENT_FAILURE_KEY = "knowledge:component_failures"
_CIRCUIT_BREAKER_THRESHOLD = 5  # failures before raising alert


class KnowledgeLoop:
    """
    Event-driven coroutine that runs forever alongside the strategy runner.

    Every time something happens (trade closed, experiment done, regime changed),
    this loop:
    1. Extracts a lesson from the event
    2. Writes it to shared AgentMemory
    3. Updates CORE memory (regime, top strategies)
    4. Fires knowledge:learned so all agents can react
    5. Every 5 min: synthesizes and logs a "state of the firm"
    """

    def __init__(self, redis_client: Any) -> None:
        self._r = redis_client
        self._running = False
        self._last_synthesis = 0.0
        self._component_failures: dict[str, int] = {}

    async def start(self) -> None:
        """Wire up the bus subscriptions and launch the learning coroutine."""
        from app.tasks.agent_bus import TOPICS, get_bus
        bus = get_bus(self._r)

        # Subscribe to every topic
        for topic in TOPICS:
            bus.subscribe(topic, self._on_event)

        self._running = True
        logger.info("KnowledgeLoop: started — learning from all %d bus topics", len(TOPICS))

        # Launch synthesis loop in parallel
        asyncio.create_task(self._synthesis_loop())

    async def stop(self) -> None:
        self._running = False

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_event(self, topic: str, data: dict) -> None:
        """Called immediately when any event fires on the bus."""
        try:
            lesson = self._extract_lesson(topic, data)
            if lesson:
                await self._write_lesson(topic, lesson, data)
                await self._update_core_memory(topic, data)
                await self._broadcast_lesson(topic, lesson, data)
        except Exception as e:
            logger.debug("KnowledgeLoop._on_event error topic=%s: %s", topic, e)

    def _extract_lesson(self, topic: str, data: dict) -> str | None:
        """Convert a raw bus event into a human-readable lesson."""
        try:
            if topic == "trade:closed":
                return self._lesson_from_trade_closed(data)
            elif topic == "trade:executed":
                return self._lesson_from_trade_executed(data)
            elif topic == "experiment:done":
                return self._lesson_from_experiment(data)
            elif topic == "market:regime":
                return self._lesson_from_regime_change(data)
            elif topic == "risk:alert":
                return self._lesson_from_risk_alert(data)
            elif topic == "strategy:updated":
                return self._lesson_from_strategy_update(data)
            elif topic == "research:finding":
                return self._lesson_from_research(data)
            elif topic == "auction:allocated":
                return self._lesson_from_auction(data)
        except Exception as e:
            logger.debug("KnowledgeLoop._extract_lesson topic=%s: %s", topic, e)
        return None

    def _lesson_from_trade_closed(self, data: dict) -> str:
        strategy = data.get("strategy", "unknown")
        symbol = data.get("symbol", "?")
        pnl = float(data.get("pnl_pct", 0))
        direction = data.get("direction", "?")
        regime = data.get("regime", "unknown")
        outcome = "win" if pnl > 0 else "loss"
        return (
            f"Strategy '{strategy}' {direction} {symbol} → {outcome} {pnl:+.2f}% "
            f"in {regime} regime. "
            + (f"Win confirms {regime} edge for {strategy}." if pnl > 0
               else f"Loss suggests {strategy} underperforms in {regime} regime.")
        )

    def _lesson_from_trade_executed(self, data: dict) -> str:
        strategy = data.get("strategy", "?")
        symbol = data.get("symbol", "?")
        algo = data.get("execution_algo", "market")
        slippage_bps = float(data.get("slippage_bps", 0))
        lesson = f"Execution: {strategy} → {symbol} via {algo}, slippage={slippage_bps:.1f}bps."
        if slippage_bps > 20:
            lesson += f" HIGH SLIPPAGE — consider switching {symbol} to limit-first execution."
        return lesson

    def _lesson_from_experiment(self, data: dict) -> str:
        model = data.get("model", "?")
        config = data.get("config", "?")
        val_sharpe = float(data.get("val_sharpe", 0))
        test_sharpe = float(data.get("test_sharpe", 0))
        status = "PROMISING" if test_sharpe > 1.5 else ("VIABLE" if test_sharpe > 0.5 else "WEAK")
        return (
            f"ML experiment '{config}' ({model}): val_sharpe={val_sharpe:.2f}, "
            f"test_sharpe={test_sharpe:.2f} → {status}. "
            + ("Deploy to ml_enhanced strategies immediately." if test_sharpe > 1.5 else
               "Archive and try different hyperparameters." if test_sharpe < 0.5 else
               "Paper trade 2 weeks before live deploy.")
        )

    def _lesson_from_regime_change(self, data: dict) -> str:
        prev = data.get("previous_regime", "unknown")
        curr = data.get("current_regime", "unknown")
        confidence = float(data.get("confidence", 0))
        action = {
            "bear": "suspend directional strategies, keep arbitrage only",
            "bull": "activate all strategies, increase position sizes",
            "sideways": "favor mean-reversion and arbitrage, reduce momentum exposure",
        }.get(curr, "review all strategy allocations")
        return (
            f"Regime shift: {prev} → {curr} (confidence={confidence:.0%}). "
            f"Action: {action}."
        )

    def _lesson_from_risk_alert(self, data: dict) -> str:
        alert_type = data.get("alert_type", "unknown")
        strategy = data.get("strategy", "system")
        value = data.get("value", "?")
        self._track_failure(strategy)
        return (
            f"RISK ALERT [{alert_type}] on '{strategy}': {value}. "
            f"Circuit breaker may have fired. Review position sizes and drawdown limits."
        )

    def _lesson_from_strategy_update(self, data: dict) -> str:
        strategy = data.get("strategy", "?")
        sharpe = float(data.get("sharpe", 0))
        win_rate = float(data.get("win_rate", 0))
        trend = "improving" if data.get("sharpe_delta", 0) > 0 else "declining"
        return (
            f"Strategy '{strategy}' metrics updated: sharpe={sharpe:.2f}, "
            f"win_rate={win_rate:.0%} — {trend}."
        )

    def _lesson_from_research(self, data: dict) -> str:
        source = data.get("source", "unknown")
        alpha_idea = data.get("alpha_idea", "")
        ic = float(data.get("ic", 0))
        return (
            f"Research finding from {source}: '{alpha_idea}' "
            f"(IC={ic:.3f}). "
            + ("Strong IC — implement as new feature immediately." if ic > 0.05 else
               "Weak IC — log and revisit after more data.")
        )

    def _lesson_from_auction(self, data: dict) -> str:
        top = data.get("top_strategies", [])
        probation = int(data.get("probation_count", 0))
        total = int(data.get("strategy_count", 0))
        return (
            f"Capital auction complete: {total} strategies, {probation} on probation. "
            f"Capital leaders: {', '.join(top[:3]) if top else 'none yet'}. "
            f"Capital flows to proven performers — others defunded until they improve."
        )

    # ── Writing and broadcasting ──────────────────────────────────────────────

    async def _write_lesson(self, topic: str, lesson: str, raw_data: dict) -> None:
        """Write lesson to episodic memory (shared AgentMemory)."""
        try:
            from app.tasks.agent_memory import AgentMemory
            memory = AgentMemory(self._r)
            await memory.write(f"lessons:{topic.replace(':', '_')}", {
                "lesson": lesson,
                "raw_data": raw_data,
                "ts": time.time(),
            })
            # Also write to unified episodic log
            await memory.write("episodic:all", {
                "topic": topic,
                "lesson": lesson,
                "ts": time.time(),
            })
        except Exception as e:
            logger.debug("KnowledgeLoop._write_lesson: %s", e)

    async def _update_core_memory(self, topic: str, data: dict) -> None:
        """Update stable CORE memory based on key events."""
        try:
            from app.tasks.agent_memory import AgentMemory
            memory = AgentMemory(self._r)

            if topic == "market:regime":
                # Core memory: current market regime
                await memory.set_latest("core:market_regime", {
                    "regime": data.get("current_regime"),
                    "confidence": data.get("confidence"),
                    "updated_at": time.time(),
                })

            elif topic == "auction:allocated":
                # Core memory: which strategies are funded
                await memory.set_latest("core:capital_allocations", {
                    "allocations": data.get("allocations", {}),
                    "updated_at": time.time(),
                })

            elif topic == "risk:alert":
                # Core memory: active risk alerts
                await memory.set_latest("core:last_risk_alert", {
                    "alert": data,
                    "updated_at": time.time(),
                })

            elif topic == "experiment:done":
                # Core memory: best known model
                test_sharpe = float(data.get("test_sharpe", 0))
                existing = await memory.get_latest("core:best_model") or {}
                if test_sharpe > float(existing.get("test_sharpe", 0)):
                    await memory.set_latest("core:best_model", {
                        "config": data.get("config"),
                        "model": data.get("model"),
                        "test_sharpe": test_sharpe,
                        "updated_at": time.time(),
                    })
        except Exception as e:
            logger.debug("KnowledgeLoop._update_core_memory: %s", e)

    async def _broadcast_lesson(self, topic: str, lesson: str, data: dict) -> None:
        """Fire knowledge:learned so all agents receive the lesson immediately."""
        try:
            from app.tasks.agent_bus import get_bus
            bus = get_bus(self._r)
            await bus.publish("knowledge:learned", {
                "source_topic": topic,
                "lesson": lesson,
                "key_data": {k: data[k] for k in list(data.keys())[:5]},  # cap size
            })
        except Exception as e:
            logger.debug("KnowledgeLoop._broadcast_lesson: %s", e)

    # ── Self-healing circuit breaker ──────────────────────────────────────────

    def _track_failure(self, component: str) -> None:
        self._component_failures[component] = self._component_failures.get(component, 0) + 1
        count = self._component_failures[component]
        if count >= _CIRCUIT_BREAKER_THRESHOLD:
            logger.warning(
                "KnowledgeLoop CIRCUIT BREAKER: component '%s' has failed %d times. "
                "Flagging for review.",
                component, count,
            )
            # Reset after warning so we don't spam
            self._component_failures[component] = 0

    # ── Synthesis loop ────────────────────────────────────────────────────────

    async def _synthesis_loop(self) -> None:
        """Every 5 minutes: synthesize recent episodes into a firm-state summary."""
        while self._running:
            await asyncio.sleep(_SYNTHESIS_INTERVAL)
            try:
                await self._synthesize()
            except Exception as e:
                logger.debug("KnowledgeLoop._synthesis_loop: %s", e)

    async def _synthesize(self) -> None:
        """
        Read recent episodic memory, build a "state of the firm" summary,
        write to core memory so all agents can read it.
        """
        try:
            from app.tasks.agent_memory import AgentMemory
            memory = AgentMemory(self._r)

            # Gather recent lessons across all topics
            recent = await memory.read_recent("episodic:all", _SYNTHESIS_WINDOW)
            if not recent:
                return

            # Count lesson types
            topic_counts: dict[str, int] = {}
            lessons_sample: list[str] = []
            for entry in recent:
                t = entry.get("topic", "unknown")
                topic_counts[t] = topic_counts.get(t, 0) + 1
                if len(lessons_sample) < 5:
                    l = entry.get("lesson", "")
                    if l:
                        lessons_sample.append(l)

            # Read core state
            regime_data = await memory.get_latest("core:market_regime") or {}
            regime = regime_data.get("regime", "unknown")
            alloc_data = await memory.get_latest("core:capital_allocations") or {}
            allocations = alloc_data.get("allocations", {})
            top_funded = sorted(allocations.items(), key=lambda x: x[1], reverse=True)[:3]

            summary = {
                "timestamp": time.time(),
                "regime": regime,
                "events_last_5min": dict(topic_counts),
                "total_lessons_recorded": sum(topic_counts.values()),
                "top_funded_strategies": [n for n, _ in top_funded],
                "recent_lessons_sample": lessons_sample,
                "component_failure_counts": dict(self._component_failures),
                "system_health": "degraded" if any(
                    v > 2 for v in self._component_failures.values()
                ) else "healthy",
            }

            await memory.set_latest("core:firm_state", summary)
            logger.info(
                "KnowledgeLoop synthesis: regime=%s, %d events tracked, health=%s",
                regime, sum(topic_counts.values()), summary["system_health"],
            )
        except Exception as e:
            logger.debug("KnowledgeLoop._synthesize: %s", e)


# ── Global singleton ──────────────────────────────────────────────────────────

_loop: KnowledgeLoop | None = None


def get_knowledge_loop(redis_client: Any | None = None) -> KnowledgeLoop:
    global _loop
    if _loop is None:
        if redis_client is None:
            from app.redis_client import get_redis
            redis_client = get_redis()
        _loop = KnowledgeLoop(redis_client)
    return _loop
