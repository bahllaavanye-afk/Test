"""
Redis pub/sub message bus for agent-to-agent collaboration.

Enhanced with:
- Token budget tracking per agent (daily quota)
- Agent-to-agent help requests with budget checking
- Cross-asset GNN signal broadcasting
- Signal stream persistence for replay
"""
import json
import uuid
from datetime import UTC, datetime

from app.utils.logging import logger

CHANNELS = {
    "strategy": "agent:findings:strategy",
    "ml":       "agent:findings:ml",
    "risk":     "agent:findings:risk",
    "p0":       "agent:alerts:p0",
    "tasks":    "agent:tasks",
    "signals":  "agent:signals:broadcast",  # GNN cross-asset signals
    "slack":    "agent:slack:outbound",      # Slack notifications
}

# Agent roster — any agent not listed gets "default" budget
AGENT_ROSTER = [
    "strategy_agent", "ml_agent", "risk_agent", "research_agent",
    "alpha_miner", "backtest_agent", "ui_agent", "coordinator",
    "modeling_engineer", "research_scientist", "algo_agent",
]


class AgentBus:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def publish(self, channel: str, message: dict, from_agent: str = "system") -> None:
        full_channel = CHANNELS.get(channel, channel)
        payload = json.dumps({
            "id": str(uuid.uuid4()),
            "from": from_agent,
            "channel": channel,
            "content": message,
            "ts": datetime.now(UTC).isoformat(),
        })
        if self.redis:
            try:
                await self.redis.publish(full_channel, payload)
            except Exception as e:
                logger.warning("AgentBus.publish failed", channel=channel, error=str(e))

    async def post_task(self, target_agent: str, task: dict, from_agent: str = "system") -> str:
        task_id = str(uuid.uuid4())
        entry = json.dumps({
            "task_id": task_id,
            "target": target_agent,
            "task": task,
            "from": from_agent,
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
        })
        if self.redis:
            try:
                await self.redis.lpush(f"agent:taskqueue:{target_agent}", entry)
                await self.redis.expire(f"agent:taskqueue:{target_agent}", 86400)
            except Exception as e:
                logger.warning("AgentBus.post_task failed", target=target_agent, error=str(e))
        return task_id

    async def claim_task(self, agent_name: str) -> dict | None:
        if not self.redis:
            return None
        try:
            raw = await self.redis.rpop(f"agent:taskqueue:{agent_name}")
        except Exception:
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def post_finding(
        self,
        channel: str,
        summary: str,
        details: dict,
        from_agent: str,
        priority: int = 2,
    ) -> None:
        await self.publish(channel, {
            "summary": summary,
            "details": details,
            "priority": priority,
        }, from_agent=from_agent)

    async def post_finding_with_budget(
        self,
        channel: str,
        summary: str,
        details: dict,
        from_agent: str,
        tokens_used: int = 0,
        priority: int = 2,
    ) -> None:
        """post_finding + records token spend against the agent's daily budget."""
        try:
            from app.tasks.token_budget import get_token_budget
            await get_token_budget().record_spend(from_agent, tokens_used)
        except Exception:
            pass
        await self.post_finding(channel, summary, details, from_agent, priority)

    async def request_help(
        self,
        from_agent: str,
        to_agent: str,
        task: dict,
        token_budget: int = 5_000,
    ) -> str:
        """
        Agent-to-agent collaboration: one agent requests help from another.
        Checks the target agent's budget before posting. Returns task_id.
        """
        try:
            from app.tasks.token_budget import get_token_budget
            budget_mgr = get_token_budget()
            if not await budget_mgr.can_spend(to_agent, token_budget):
                logger.warning(
                    "AgentBus.request_help: target over budget",
                    from_agent=from_agent,
                    to_agent=to_agent,
                    token_budget=token_budget,
                )
                # Still post — budget enforcement is advisory, not hard-stop
        except Exception:
            pass

        task["_requested_by"] = from_agent
        task["_token_budget"] = token_budget
        return await self.post_task(to_agent, task, from_agent=from_agent)

    async def broadcast_signal(self, signal: dict, from_agent: str = "coordinator") -> None:
        """
        Broadcast a cross-asset ML signal to all subscriber agents.
        Used by GNN coordinator to fan-out enhanced signals to strategy agents.
        Persists to a bounded Redis list for replay/audit.
        """
        enriched = {**signal, "from": from_agent, "broadcast_ts": datetime.now(UTC).isoformat()}
        await self.publish("signals", enriched, from_agent=from_agent)
        if self.redis:
            try:
                await self.redis.lpush("agent:signals:stream", json.dumps(enriched))
                await self.redis.ltrim("agent:signals:stream", 0, 4999)  # keep last 5000
                await self.redis.expire("agent:signals:stream", 604800)  # 7 days
            except Exception as e:
                logger.debug("AgentBus.broadcast_signal stream write failed", error=str(e))

    async def get_recent_signals(self, limit: int = 50) -> list[dict]:
        """Retrieve recent cross-asset signals from the stream."""
        if not self.redis:
            return []
        try:
            raw_list = await self.redis.lrange("agent:signals:stream", 0, limit - 1)
            result = []
            for raw in (raw_list or []):
                try:
                    result.append(json.loads(raw))
                except Exception:
                    pass
            return result
        except Exception:
            return []

    async def slack_notify(self, message: str, from_agent: str, level: str = "info") -> None:
        """Route a Slack notification through the bus → slack_handler picks it up."""
        await self.publish("slack", {
            "text": message,
            "level": level,
        }, from_agent=from_agent)

    async def get_agent_status(self) -> list[dict]:
        """Return token budget status for all known agents."""
        try:
            from app.tasks.token_budget import get_token_budget
            return await get_token_budget().all_usage()
        except Exception:
            return []


# ── Global singleton ──────────────────────────────────────────────────────────

_bus: AgentBus | None = None


def get_bus(redis_client=None) -> AgentBus:
    global _bus
    if _bus is None:
        if redis_client is None:
            try:
                from app.redis_client import get_redis
                redis_client = get_redis()
            except Exception:
                redis_client = None
        _bus = AgentBus(redis_client)
    return _bus
