"""Real-time price WebSocket endpoint."""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Wildcard topic used for subscribers that want all symbols
PRICES_ALL_TOPIC = "prices:*"


@router.websocket("/ws/prices")
async def prices_ws_all(websocket: WebSocket):
    """Subscribe to all price updates across all symbols."""
    await manager.connect(websocket, PRICES_ALL_TOPIC)
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except Exception as exc:
                logger.warning("prices_ws_all receive error: %s", exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, PRICES_ALL_TOPIC)


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str):
    topic = f"prices:{symbol}"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except Exception as exc:
                logger.warning("prices_ws receive error for %s: %s", symbol, exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, topic)
