"""Real-time agent activity log WebSocket endpoint."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from jose import JWTError

from app.utils.security import decode_token
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()

AGENT_LOGS_TOPIC = "agent_logs"


@router.websocket("/ws/agent-logs")
async def ws_agent_logs(
    websocket: WebSocket,
    token: str = Query(...),
) -> None:
    """Real-time stream of agent activity log entries. Requires a valid JWT token."""
    # Validate the token before accepting the connection
    try:
        payload = decode_token(token)
        user_id: str | None = payload.get("sub")
        if not user_id or payload.get("type") != "access":
            await websocket.close(code=4001, reason="Unauthorized")
            return
    except JWTError:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    topic = f"{AGENT_LOGS_TOPIC}:{user_id}"
    # Also subscribe to the broadcast channel so all users get real-time events
    await manager.connect(websocket, AGENT_LOGS_TOPIC)
    try:
        while True:
            try:
                # Heartbeat every 30 seconds; also keep connection alive by draining messages
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
            except Exception as exc:
                logger.warning("ws_agent_logs receive error for user %s: %s", user_id, exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, AGENT_LOGS_TOPIC)
