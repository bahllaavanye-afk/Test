"""Strategy signal alerts WebSocket endpoint."""
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket):
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
