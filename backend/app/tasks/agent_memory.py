"""
Persistent agent memory via Redis.
Each agent has working memory (24h TTL), long-term memory (permanent), and episodic log (30d).
"""
import json
import hashlib
from datetime import datetime
from typing import Any, Optional


class AgentMemory:
    def __init__(self, agent_name: str, redis_client):
        self.agent_name = agent_name
        self.redis = redis_client
        self._prefix = f"agent:memory:{agent_name}"

    def _key(self, memory_type: str, key: str) -> str:
        return f"{self._prefix}:{memory_type}:{key}"

    async def remember(self, key: str, value: Any, memory_type: str = "working") -> None:
        full_key = self._key(memory_type, key)
        payload = json.dumps({"value": value, "updated_at": datetime.utcnow().isoformat()})
        if memory_type == "working":
            await self.redis.set(full_key, payload, ex=86400)  # 24h TTL
        elif memory_type == "long_term":
            await self.redis.set(full_key, payload)  # no TTL
        # episodic handled separately

    async def recall(self, key: str, memory_type: str = "working") -> Any:
        full_key = self._key(memory_type, key)
        raw = await self.redis.get(full_key)
        if raw is None:
            return None
        try:
            return json.loads(raw)["value"]
        except Exception:
            return None

    async def log_episode(self, outcome: dict) -> None:
        key = f"{self._prefix}:episodes"
        entry = json.dumps({**outcome, "ts": datetime.utcnow().isoformat()})
        await self.redis.lpush(key, entry)
        await self.redis.ltrim(key, 0, 999)   # keep last 1000 episodes
        await self.redis.expire(key, 86400 * 30)  # 30 day TTL

    async def get_recent_episodes(self, n: int = 10) -> list[dict]:
        key = f"{self._prefix}:episodes"
        raw_list = await self.redis.lrange(key, 0, n - 1)
        episodes = []
        for raw in (raw_list or []):
            try:
                episodes.append(json.loads(raw))
            except Exception:
                pass
        return episodes

    async def get_context(self, max_chars: int = 2000) -> str:
        """Returns formatted memory string to prepend to LLM prompts."""
        lines = [f"[AgentMemory: {self.agent_name}]"]
        # Recent episodes
        episodes = await self.get_recent_episodes(5)
        if episodes:
            lines.append("Recent outcomes:")
            for ep in episodes:
                lines.append(f"  - {ep.get('ts', '')[:10]}: {ep.get('summary', str(ep))[:120]}")
        context = "\n".join(lines)
        return context[:max_chars]
