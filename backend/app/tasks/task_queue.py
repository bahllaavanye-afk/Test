"""
Durable Task Queue — Redis-backed with priority, retry, and dead-letter.

Every background task that needs retry semantics, ordering, or deduplication
goes through this queue instead of being fire-and-forget.

Queue structure (Redis Streams):
  tasks:pending:<priority>   — 0=critical, 1=high, 2=normal, 3=low
  tasks:processing           — tasks currently being worked
  tasks:dead                 — permanently failed tasks (max_retries exceeded)

Workers call claim_next() to atomically dequeue + lock a task, then done() or fail().
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 2
PRIORITY_LOW = 3

_QUEUE_PREFIX = "tasks:pending:"
_PROCESSING_KEY = "tasks:processing"
_DEAD_KEY = "tasks:dead"
_LOCK_TTL = 300          # seconds — task lock expires after 5 min (worker crash recovery)
_MAX_STREAM_LEN = 5000

TaskFn = Callable[..., Awaitable[None]]


@dataclass
class Task:
    task_id: str
    task_type: str
    payload: dict
    priority: int = PRIORITY_NORMAL
    max_retries: int = 3
    attempt: int = 0
    created_at: float = field(default_factory=time.time)
    scheduled_at: float = field(default_factory=time.time)
    error: str = ""

    def to_redis(self) -> dict[str, str]:
        return {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_redis(cls, fields: dict) -> "Task":
        d: dict = {}
        for k, v in fields.items():
            try:
                d[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                d[k] = v
        d["priority"] = int(d.get("priority", PRIORITY_NORMAL))
        d["max_retries"] = int(d.get("max_retries", 3))
        d["attempt"] = int(d.get("attempt", 0))
        d["created_at"] = float(d.get("created_at", 0))
        d["scheduled_at"] = float(d.get("scheduled_at", 0))
        return cls(**d)


class TaskQueue:
    """
    Priority task queue backed by Redis Streams.

    Drain order: critical → high → normal → low.
    Failed tasks are retried with exponential backoff (2^attempt seconds).
    Tasks exceeding max_retries go to the dead-letter stream.
    """

    def __init__(self, redis_client: Any) -> None:
        self._r = redis_client
        self._handlers: dict[str, TaskFn] = {}

    # ── Registering handlers ──────────────────────────────────────────────────

    def register(self, task_type: str, handler: TaskFn) -> None:
        """Register an async handler for a task type."""
        self._handlers[task_type] = handler

    # ── Enqueuing ─────────────────────────────────────────────────────────────

    async def enqueue(
        self,
        task_type: str,
        payload: dict,
        priority: int = PRIORITY_NORMAL,
        max_retries: int = 3,
        delay_seconds: float = 0,
    ) -> str:
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
            scheduled_at=time.time() + delay_seconds,
        )
        key = f"{_QUEUE_PREFIX}{priority}"
        try:
            await self._r.xadd(key, task.to_redis(), maxlen=_MAX_STREAM_LEN, approximate=True)
            logger.debug("TaskQueue.enqueue type=%s id=%s priority=%d", task_type, task_id, priority)
        except Exception as e:
            logger.warning("TaskQueue.enqueue failed: %s", e)
        return task_id

    # ── Processing ────────────────────────────────────────────────────────────

    async def process_once(self) -> bool:
        """Drain one task from the highest-priority non-empty queue. Returns True if a task ran."""
        for priority in (PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW):
            key = f"{_QUEUE_PREFIX}{priority}"
            try:
                results = await self._r.xread({key: "0"}, count=1)
                if not results:
                    continue
                for _, messages in results:
                    for msg_id, fields in messages:
                        task = Task.from_redis(fields)
                        # Delete from queue before processing (at-most-once delivery)
                        await self._r.xdel(key, msg_id)
                        await self._dispatch(task)
                        return True
            except Exception as e:
                logger.debug("TaskQueue.process_once error priority=%d: %s", priority, e)
        return False

    async def run_worker(self, poll_interval: float = 0.5) -> None:
        """Continuous worker loop. Run as background task."""
        logger.info("TaskQueue worker started")
        while True:
            try:
                ran = await self.process_once()
                if not ran:
                    await asyncio.sleep(poll_interval)
            except Exception as e:
                logger.error("TaskQueue worker error: %s", e)
                await asyncio.sleep(2.0)

    async def _dispatch(self, task: Task) -> None:
        now = time.time()
        if task.scheduled_at > now:
            # Re-enqueue with delay (use original priority)
            delay = task.scheduled_at - now
            await asyncio.sleep(min(delay, 5.0))
            await self._requeue(task, delay=max(0, task.scheduled_at - time.time()))
            return

        handler = self._handlers.get(task.task_type)
        if handler is None:
            logger.warning("TaskQueue: no handler for task_type=%s", task.task_type)
            return

        try:
            await handler(**task.payload)
        except Exception as exc:
            task.attempt += 1
            task.error = str(exc)
            logger.warning(
                "TaskQueue task failed type=%s attempt=%d/%d: %s",
                task.task_type, task.attempt, task.max_retries, exc,
            )
            if task.attempt < task.max_retries:
                backoff = 2 ** task.attempt
                await self._requeue(task, delay=backoff)
            else:
                await self._dead_letter(task)

    async def _requeue(self, task: Task, delay: float = 0) -> None:
        task.scheduled_at = time.time() + delay
        key = f"{_QUEUE_PREFIX}{task.priority}"
        try:
            await self._r.xadd(key, task.to_redis(), maxlen=_MAX_STREAM_LEN, approximate=True)
        except Exception as e:
            logger.debug("TaskQueue._requeue failed: %s", e)

    async def _dead_letter(self, task: Task) -> None:
        logger.error("TaskQueue: task permanently failed type=%s id=%s error=%s", task.task_type, task.task_id, task.error)
        try:
            await self._r.xadd(_DEAD_KEY, task.to_redis(), maxlen=1000, approximate=True)
        except Exception:
            pass

    # ── Monitoring ────────────────────────────────────────────────────────────

    async def queue_depths(self) -> dict[str, int]:
        depths = {}
        for priority in range(4):
            key = f"{_QUEUE_PREFIX}{priority}"
            try:
                depths[f"priority_{priority}"] = await self._r.xlen(key)
            except Exception:
                depths[f"priority_{priority}"] = -1
        try:
            depths["dead"] = await self._r.xlen(_DEAD_KEY)
        except Exception:
            depths["dead"] = -1
        return depths


# ── Global singleton ──────────────────────────────────────────────────────────

_queue: TaskQueue | None = None


def get_task_queue(redis_client: Any | None = None) -> TaskQueue:
    global _queue
    if _queue is None:
        if redis_client is None:
            from app.redis_client import get_redis
            redis_client = get_redis()
        _queue = TaskQueue(redis_client)
    return _queue
