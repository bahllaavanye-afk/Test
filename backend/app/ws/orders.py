"""Real-time order status WebSocket endpoint."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

router = APIRouter()


@router.websocket("/ws/orders")
async def orders_ws(websocket: WebSocket):
    topic = "orders"
    await manager.connect(websocket, topic)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, topic)
