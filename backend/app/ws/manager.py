"""WebSocket connection manager with topic-based pub/sub."""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from fastapi import WebSocket
from app.utils.logging import logger


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, topic: str) -> None:
        await websocket.accept()
        self._connections[topic].add(websocket)
        logger.info(
            "WebSocket connected",
            topic=topic,
            total=len(self._connections[topic]),
        )

    def disconnect(self, websocket: WebSocket, topic: str) -> None:
        self._connections[topic].discard(websocket)

    def _targets_for(self, topic: str) -> set[WebSocket]:
        """All sockets that should receive a broadcast to ``topic``.

        This includes exact-topic subscribers plus any wildcard subscriber registered
        under ``"<prefix>:*"`` — e.g. ``/ws/prices`` (all symbols) subscribes to
        ``"prices:*"`` and must receive every concrete ``"prices:{symbol}"`` update.
        Without this, the all-symbols ticker silently received nothing.
        """
        targets = set(self._connections.get(topic, set()))
        if ":" in topic and not topic.endswith(":*"):
            prefix = topic.rsplit(":", 1)[0]
            targets |= self._connections.get(f"{prefix}:*", set())
        return targets

    async def broadcast(self, topic: str, data: dict) -> None:
        """Send ``data`` to all subscribers of ``topic``.

        Structured logging at INFO level captures:
        - ``signal_count``: number of sockets the message was sent to.
        - ``exec_time_ms``: time taken to perform the broadcast.
        - ``pnl``: optional profit & loss metric from ``data`` if present.
        """
        start_time = time.perf_counter()
        message = json.dumps(data)
        dead = set()
        targets = self._targets_for(topic)
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)

        # Remove dead sockets from all topics they may belong to.
        if dead:
            for sockets in self._connections.values():
                sockets -= dead

        exec_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Broadcast sent",
            topic=topic,
            signal_count=len(targets),
            exec_time_ms=exec_time_ms,
            pnl=data.get("pnl"),
        )

    async def broadcast_all(self, data: dict) -> None:
        for topic in list(self._connections.keys()):
            await self.broadcast(topic, data)


manager = ConnectionManager()