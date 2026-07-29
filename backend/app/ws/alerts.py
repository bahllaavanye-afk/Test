"""Strategy signal alerts WebSocket endpoint."""
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/alerts")
async def alerts_ws(websocket: Optional[WebSocket]) -> None:
    """WebSocket endpoint for real‑time alerts.

    Handles edge cases such as a ``None`` websocket, empty messages,
    and ensures clean disconnection even when unexpected errors occur.
    """
    if websocket is None:
        logger.warning("alerts_ws called with None websocket; aborting.")
        return

    topic = "alerts"

    # Guard against manager.connect failures (e.g., invalid topic or websocket)
    try:
        await manager.connect(websocket, topic)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to connect websocket to manager: %s", exc)
        return

    try:
        while True:
            try:
                message = await websocket.receive_text()
                # Skip processing for empty payloads to avoid off‑by‑one logic issues
                if not message:
                    logger.debug("Received empty alert message; ignoring.")
                    continue
                # Placeholder for future message handling logic
            except WebSocketDisconnect:
                # Normal disconnection flow; break the loop to clean up
                break
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("alerts_ws receive error: %s", exc)
                # Continue listening after logging; break only on critical failures
                break
    finally:
        # Ensure the websocket is always removed from the manager
        try:
            manager.disconnect(websocket, topic)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error during websocket disconnect: %s", exc)