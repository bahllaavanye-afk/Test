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
    if websocket is None:
        logger.error("prices_ws_all called with None websocket")
        return
    try:
        await manager.connect(websocket, PRICES_ALL_TOPIC)
    except Exception as exc:
        logger.error("Failed to connect manager for all prices: %s", exc)
        return
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
        try:
            manager.disconnect(websocket, PRICES_ALL_TOPIC)
        except Exception as exc:
            logger.error("Error during disconnect for all prices: %s", exc)


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str):
    """Subscribe to price updates for a specific symbol."""
    if websocket is None:
        logger.error("prices_ws called with None websocket")
        return
    if not symbol:
        logger.warning("prices_ws called with empty or None symbol")
        # Close the connection gracefully if possible
        try:
            await websocket.close()
        except Exception:
            pass
        return
    topic = f"prices:{symbol}"
    try:
        await manager.connect(websocket, topic)
    except Exception as exc:
        logger.error("Failed to connect manager for symbol %s: %s", symbol, exc)
        return
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
        try:
            manager.disconnect(websocket, topic)
        except Exception as exc:
            logger.error("Error during disconnect for symbol %s: %s", symbol, exc)