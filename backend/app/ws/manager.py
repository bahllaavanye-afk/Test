"""WebSocket connection manager with topic-based pub/sub."""
from __future__ import annotations

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

    async def broadcast(self, topic: str, data: dict) -> None:
        message = json.dumps(data)
        dead = set()
        for ws in self._connections.get(topic, set()):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections[topic].discard(ws)

    async def broadcast_all(self, data: dict) -> None:
        for topic in list(self._connections.keys()):
            await self.broadcast(topic, data)


manager = ConnectionManager()
