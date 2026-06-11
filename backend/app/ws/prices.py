"""Real-time price WebSocket endpoint."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

router = APIRouter()

# Wildcard topic used for subscribers that want all symbols
PRICES_ALL_TOPIC = "prices:*"


@router.websocket("/ws/prices")
async def prices_ws_all(websocket: WebSocket):
    """Subscribe to all price updates across all symbols."""
    await manager.connect(websocket, PRICES_ALL_TOPIC)
    try:
        while True:
            await websocket.receive_text()  # keep alive / ping handling
    except WebSocketDisconnect:
        manager.disconnect(websocket, PRICES_ALL_TOPIC)


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str):
    topic = f"prices:{symbol}"
    await manager.connect(websocket, topic)
    try:
        while True:
            await websocket.receive_text()  # keep alive / ping handling
    except WebSocketDisconnect:
        manager.disconnect(websocket, topic)
