"""Real-time price WebSocket endpoint."""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Constants
PRICES_WS_ENDPOINT = "/ws/prices"
PRICES_WS_SYMBOL_ENDPOINT = "/ws/prices/{symbol}"
PRICES_ALL_TOPIC = "prices:*"
LOG_MSG_ALL_RECEIVE_ERROR = "prices_ws_all receive error: %s"
LOG_MSG_RECEIVE_ERROR = "prices_ws receive error for %s: %s"


@router.websocket(PRICES_WS_ENDPOINT)
async def prices_ws_all(websocket: WebSocket):
    """Subscribe to all price updates across all symbols."""
    await manager.connect(websocket, PRICES_ALL_TOPIC)
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except Exception as exc:
                logger.warning(LOG_MSG_ALL_RECEIVE_ERROR, exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, PRICES_ALL_TOPIC)


@router.websocket(PRICES_WS_SYMBOL_ENDPOINT)
async def prices_ws(websocket: WebSocket, symbol: str):
    topic = f"prices:{symbol}"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except Exception as exc:
                logger.warning(LOG_MSG_RECEIVE_ERROR, symbol, exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, topic)