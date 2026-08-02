"""Real-time price WebSocket endpoint."""
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Wildcard topic used for subscribers that want all symbols
PRICES_ALL_TOPIC = "prices:*"


async def _handle_ws(
    websocket: WebSocket,
    topic: str,
    symbol: Optional[str] = None,
) -> None:
    """Core WebSocket handler.

    Keeps the connection alive by consuming incoming messages (e.g., pings) and
    ensures proper registration and cleanup with the ``manager``.

    Args:
        websocket: The FastAPI WebSocket instance.
        topic: Subscription topic for the manager.
        symbol: Optional symbol name for logging context.
    """
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except Exception as exc:  # pragma: no cover
                log_msg = (
                    f"prices_ws receive error for {symbol}: {exc}"
                    if symbol
                    else f"prices_ws_all receive error: {exc}"
                )
                logger.warning(log_msg)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, topic)


@router.websocket("/ws/prices")
async def prices_ws_all(websocket: WebSocket) -> None:
    """Subscribe to all price updates across all symbols."""
    await _handle_ws(websocket, PRICES_ALL_TOPIC)


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str) -> None:
    """Subscribe to price updates for a specific symbol."""
    topic = f"prices:{symbol}"
    await _handle_ws(websocket, topic, symbol)