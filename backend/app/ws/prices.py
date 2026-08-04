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
    # Edge‑case handling: ensure a valid topic is provided
    if not topic:
        logger.error(
            "WebSocket connection attempted with empty topic",
            extra={"symbol": symbol, "topic": topic},
        )
        await websocket.close()
        return

    try:
        await manager.connect(websocket, topic)
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "Failed to register WebSocket with manager",
            extra={"symbol": symbol, "topic": topic, "error": str(exc)},
        )
        await websocket.close()
        return

    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except WebSocketDisconnect:
                # Normal disconnect; break the loop to cleanup
                break
            except Exception as exc:  # pragma: no cover
                logger.exception(
                    "Error receiving message on price WebSocket",
                    extra={"symbol": symbol, "topic": topic, "error": str(exc)},
                )
                break
    finally:
        try:
            manager.disconnect(websocket, topic)
        except Exception as exc:  # pragma: no cover
            logger.exception(
                "Error during WebSocket cleanup",
                extra={"symbol": symbol, "topic": topic, "error": str(exc)},
            )


@router.websocket("/ws/prices")
async def prices_ws_all(websocket: WebSocket) -> None:
    """Subscribe to all price updates across all symbols."""
    await _handle_ws(websocket, PRICES_ALL_TOPIC)


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str) -> None:
    """Subscribe to price updates for a specific symbol."""
    # Edge‑case handling: reject empty or None symbols
    if not symbol:
        logger.error(
            "Attempted to subscribe with empty symbol",
            extra={"symbol": symbol},
        )
        await websocket.close()
        return

    topic = f"prices:{symbol}"
    # Guard against off‑by‑one errors that could produce an empty topic suffix
    if not topic.strip():
        logger.error(
            "Generated empty topic for symbol subscription",
            extra={"symbol": symbol},
        )
        await websocket.close()
        return

    await _handle_ws(websocket, topic, symbol)