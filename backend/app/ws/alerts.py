"""WebSocket endpoint for broadcasting strategy signal alerts.

This module defines a single WebSocket route that clients can connect to in order
to receive real‑time alerts about trading strategy signals. The endpoint registers
the connection with the global WebSocket manager under the ``alerts`` topic and
keeps the connection alive until the client disconnects or an error occurs while
receiving data.
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket) -> None:
    """Handle a WebSocket connection for alert notifications.

    The connection is registered with the global ``manager`` under the
    ``alerts`` topic. Incoming messages are read and discarded; any receive
    error results in a warning and termination of the loop. On disconnect,
    the socket is removed from the manager.

    Args:
        websocket: The FastAPI ``WebSocket`` instance representing the client
            connection.
    """
    topic = "alerts"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                await websocket.receive_text()
            except Exception as exc:
                logger.warning("alerts_ws receive error: %s", exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, topic)