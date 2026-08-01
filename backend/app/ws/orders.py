"""Real-time order status WebSocket endpoint."""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


async def _receive_messages(websocket: WebSocket, topic: str) -> None:
    """Continuously receive messages from the client.

    The function loops until a receive error occurs, logging the exception
    and exiting the loop. It does not raise the exception to the caller,
    allowing the caller to handle cleanup uniformly.
    """
    while True:
        try:
            await websocket.receive_text()
        except Exception as exc:  # pragma: no cover
            logger.warning("orders_ws receive error: %s", exc)
            break


@router.websocket("/ws/orders")
async def orders_ws(websocket: WebSocket):
    """WebSocket endpoint for order updates.

    Connects the client to the ``orders`` topic, forwards incoming messages
    to ``_receive_messages``, and ensures proper disconnection handling.
    """
    topic = "orders"
    await manager.connect(websocket, topic)
    try:
        await _receive_messages(websocket, topic)
    except WebSocketDisconnect:
        # Normal disconnection; no additional action required.
        pass
    finally:
        manager.disconnect(websocket, topic)