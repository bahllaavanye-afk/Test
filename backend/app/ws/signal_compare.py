"""
WebSocket /ws/signal-compare
Streams real-time comparison between manual and ML strategy signals.
"""
import asyncio
import time

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

router = APIRouter()


def _verify_ws_token(token: str):
    """Verify a JWT token for WebSocket connections.
    Returns the decoded payload or raises an exception if invalid.
    """
    from app.utils.security import decode_token
    return decode_token(token)


@router.websocket("/ws/signal-compare")
async def ws_signal_compare(
    websocket: WebSocket,
    token: str = Query(...),
):
    try:
        _verify_ws_token(token)
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    try:
        while True:
            # Subscribe to signal comparison events from Redis
            # In production: use Redis pub/sub; here we send heartbeat
            await asyncio.sleep(30)
            await websocket.send_json({"type": "heartbeat", "ts": time.time()})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
