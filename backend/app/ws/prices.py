"""Real-time price WebSocket endpoint."""
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Wildcard topic used for subscribers that want all symbols
PRICES_ALL_TOPIC = "prices:*"


@router.websocket("/ws/prices")
async def prices_ws_all(websocket: WebSocket) -> None:
    """Subscribe to all price updates across all symbols."""
    await manager.connect(websocket, PRICES_ALL_TOPIC)
    try:
        while True:
            await websocket.receive_text()  # keep alive / ping handling
    except WebSocketDisconnect:
        # Normal disconnect; no action needed beyond cleanup
        pass
    except Exception as exc:
        logger.warning("prices_ws_all receive error: %s", exc)
    finally:
        await manager.disconnect(websocket, PRICES_ALL_TOPIC)


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str) -> None:
    """Subscribe to price updates for a specific symbol."""
    topic = f"prices:{symbol}"
    await manager.connect(websocket, topic)
    try:
        while True:
            await websocket.receive_text()  # keep alive / ping handling
    except WebSocketDisconnect:
        # Normal disconnect; cleanup will occur in finally block
        pass
    except Exception as exc:
        logger.warning("prices_ws receive error for %s: %s", symbol, exc)
    finally:
        await manager.disconnect(websocket, topic)