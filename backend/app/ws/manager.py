"""WebSocket connection manager with topic-based pub/sub."""
from __future__ import annotations
import asyncio
import json
from collections import defaultdict
from fastapi import WebSocket
from app.utils.logging import logger


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, topic: str) -> None:
        await websocket.accept()
        self._connections[topic].add(websocket)
        logger.info("WebSocket connected", topic=topic, total=len(self._connections[topic]))

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
        message = json.dumps(data)
        dead = set()
        for ws in self._targets_for(topic):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        # A dead socket may live under either the exact or the wildcard topic — purge both.
        if dead:
            for sockets in self._connections.values():
                sockets -= dead

    async def broadcast_all(self, data: dict) -> None:
        for topic in list(self._connections.keys()):
            await self.broadcast(topic, data)


manager = ConnectionManager()
