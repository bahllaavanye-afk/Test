"""WebSocket connection manager with topic-based pub/sub."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from fastapi import WebSocket

from app.utils.logging import logger


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, topic: str) -> None:
        """Accept a new websocket and register it under the given topic."""
        await websocket.accept()
        self._connections[topic].add(websocket)
        logger.info(
            "WebSocket connected",
            topic=topic,
            total=len(self._connections[topic]),
        )

    def disconnect(self, websocket: WebSocket, topic: str) -> None:
        """Remove a websocket from the subscription list of a topic."""
        self._connections[topic].discard(websocket)

    def _targets_for(self, topic: str) -> set[WebSocket]:
        """Return all sockets that should receive a broadcast for ``topic``.

        Includes:
        * Exact‑topic subscribers.
        * Wildcard subscribers of the form ``\"<prefix>:*\"`` when ``topic`` has a
          prefix (e.g. ``\"prices:BTC\"`` will also be sent to ``\"prices:*\"``).
        """
        targets = set(self._connections.get(topic, set()))
        if ":" in topic and not topic.endswith(":*"):
            prefix = topic.rsplit(":", 1)[0]
            targets |= self._connections.get(f"{prefix}:*", set())
        return targets

    async def _send_message(self, websocket: WebSocket, message: str) -> bool:
        """Send a JSON message to a single websocket.

        Returns ``True`` if the send succeeded, ``False`` otherwise.
        """
        try:
            await websocket.send_text(message)
            return True
        except Exception:
            return False

    def _cleanup_dead_sockets(self, dead: set[WebSocket]) -> None:
        """Remove dead sockets from all subscription sets."""
        for sockets in self._connections.values():
            sockets.difference_update(dead)

    async def broadcast(self, topic: str, data: dict) -> None:
        """Broadcast ``data`` to all websockets interested in ``topic``."""
        message = json.dumps(data)
        dead: set[WebSocket] = set()

        for ws in self._targets_for(topic):
            if not await self._send_message(ws, message):
                dead.add(ws)

        if dead:
            self._cleanup_dead_sockets(dead)

    async def broadcast_all(self, data: dict) -> None:
        """Broadcast ``data`` to every topic currently registered."""
        for topic in list(self._connections.keys()):
            await self.broadcast(topic, data)


manager = ConnectionManager()