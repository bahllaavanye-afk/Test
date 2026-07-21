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
    """Subscribe to all price updates across all symbols.

    The endpoint maintains the WebSocket connection solely for keep‑alive
    (ping) messages. All actual price broadcasting is handled by the
    ``manager`` instance.
    """
    await manager.connect(websocket, PRICES_ALL_TOPIC)
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except WebSocketDisconnect as exc:
                logger.info("prices_ws_all client disconnected: %s", exc)
                break
            except Exception as exc:
                logger.exception(
                    "Unexpected error in prices_ws_all receive loop for topic %s",
                    PRICES_ALL_TOPIC,
                )
                break
    finally:
        # Ensure cleanup even if an unexpected exception occurs
        manager.disconnect(websocket, PRICES_ALL_TOPIC)


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str) -> None:
    """Subscribe to price updates for a specific ``symbol``.

    The endpoint only processes ping/keep‑alive messages; price updates are
    pushed to the client via the ``manager``.
    """
    topic = f"prices:{symbol}"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except WebSocketDisconnect as exc:
                logger.info("prices_ws client disconnected for %s: %s", symbol, exc)
                break
            except Exception as exc:
                logger.exception(
                    "Unexpected error in prices_ws receive loop for symbol %s, topic %s",
                    symbol,
                    topic,
                )
                break
    finally:
        # Ensure the socket is deregistered regardless of how the loop exits
        manager.disconnect(websocket, topic)