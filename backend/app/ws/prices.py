"""Real-time price WebSocket endpoint."""
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Wildcard topic used for subscribers that want all symbols
PRICES_ALL_TOPIC = "prices:*"

# Inactivity timeout (seconds) after which the connection is closed
INACTIVITY_TIMEOUT = 300


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
                # Wait for a message with a timeout to allow early exit on inactivity
                await asyncio.wait_for(websocket.receive_text(), timeout=INACTIVITY_TIMEOUT)
            except asyncio.TimeoutError:
                # No activity within the timeout period; close connection early
                logger.debug(
                    "WebSocket inactive, closing connection",
                    extra={"symbol": symbol, "topic": topic},
                )
                break
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
    topic = f"prices:{symbol}"
    await _handle_ws(websocket, topic, symbol)