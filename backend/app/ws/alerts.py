"""Strategy signal alerts WebSocket endpoint."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

router = APIRouter()


@router.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket):
    topic = "alerts"
    await manager.connect(websocket, topic)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, topic)
