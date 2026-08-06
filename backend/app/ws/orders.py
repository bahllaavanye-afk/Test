"""Real-time order status WebSocket endpoint."""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/orders")
async def orders_ws(websocket: WebSocket):
    # Guard against None websocket (should not happen, but handle gracefully)
    if websocket is None:
        logger.error("orders_ws called with None websocket")
        return

    topic = "orders"
    # Ensure topic is a non‑empty string
    if not topic:
        logger.error("orders_ws: topic is empty; aborting connection")
        return

    # Connect the websocket; protect against unexpected errors in manager
    try:
        await manager.connect(websocket, topic)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to connect websocket to manager: %s", exc)
        return

    try:
        while True:
            try:
                data = await websocket.receive_text()
                # Handle None or empty payloads gracefully
                if not data:
                    logger.debug("Received empty message; ignoring")
                    continue
                # Process data if needed (currently a placeholder)
            except Exception as exc:
                logger.warning("orders_ws receive error: %s", exc)
                break
    except WebSocketDisconnect:
        # Normal disconnect; no action needed beyond cleanup
        pass
    finally:
        # Ensure disconnection cleanup even if an unexpected exception occurs
        try:
            manager.disconnect(websocket, topic)
        except Exception as exc:  # pragma: no cover
            logger.exception("Error during manager.disconnect: %s", exc)