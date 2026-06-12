"""
Redis pub/sub message bus for agent-to-agent collaboration.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional


CHANNELS = {
    "strategy": "agent:findings:strategy",
    "ml":       "agent:findings:ml",
    "risk":     "agent:findings:risk",
    "p0":       "agent:alerts:p0",
    "tasks":    "agent:tasks",
}


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
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        await self.redis.publish(full_channel, payload)

    async def post_task(self, target_agent: str, task: dict, from_agent: str = "system") -> str:
        task_id = str(uuid.uuid4())
        entry = json.dumps({
            "task_id": task_id,
            "target": target_agent,
            "task": task,
            "from": from_agent,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        await self.redis.lpush(f"agent:taskqueue:{target_agent}", entry)
        await self.redis.expire(f"agent:taskqueue:{target_agent}", 86400)
        return task_id

    async def claim_task(self, agent_name: str) -> Optional[dict]:
        raw = await self.redis.rpop(f"agent:taskqueue:{agent_name}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def post_finding(self, channel: str, summary: str, details: dict, from_agent: str, priority: int = 2) -> None:
        await self.publish(channel, {
            "summary": summary,
            "details": details,
            "priority": priority,
        }, from_agent=from_agent)


# ── Global singleton ──────────────────────────────────────────────────────────

_bus: AgentBus | None = None


def get_bus(redis_client=None) -> AgentBus:
    global _bus
    if _bus is None:
        if redis_client is None:
            from app.redis_client import get_redis
            redis_client = get_redis()
        _bus = AgentBus(redis_client)
    return _bus
