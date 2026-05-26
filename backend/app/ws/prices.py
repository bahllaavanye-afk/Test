"""Real-time price WebSocket endpoint."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

router = APIRouter()


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str):
    topic = f"prices:{symbol}"
    await manager.connect(websocket, topic)
    try:
        while True:
            await websocket.receive_text()  # keep alive / ping handling
    except WebSocketDisconnect:
        manager.disconnect(websocket, topic)
