"""Real-time order status WebSocket endpoint."""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/orders")
async def orders_ws(websocket: WebSocket):
    topic = "orders"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                await websocket.receive_text()
            except Exception as exc:
                logger.warning("orders_ws receive error: %s", exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, topic)
