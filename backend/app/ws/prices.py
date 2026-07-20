"""Real-time price WebSocket endpoint."""
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Wildcard topic used for subscribers that want all symbols
PRICES_ALL_TOPIC = "prices:*"


@router.websocket("/ws/prices")
async def prices_ws_all(websocket: WebSocket):
    """Subscribe to all price updates across all symbols."""
    await manager.connect(websocket, PRICES_ALL_TOPIC)
    signal_count = 0
    start_time = time.monotonic()
    logger.info(
        "prices_ws_all connection opened",
        extra={"topic": PRICES_ALL_TOPIC, "signal_count": signal_count},
    )
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
                signal_count += 1
            except Exception as exc:
                logger.warning("prices_ws_all receive error: %s", exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        duration = time.monotonic() - start_time
        # P&L is not applicable in this endpoint; placeholder set to None
        pnl = None
        logger.info(
            "prices_ws_all connection closed",
            extra={
                "topic": PRICES_ALL_TOPIC,
                "signal_count": signal_count,
                "execution_time_seconds": duration,
                "pnl": pnl,
            },
        )
        manager.disconnect(websocket, PRICES_ALL_TOPIC)


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str):
    """Subscribe to price updates for a specific symbol."""
    topic = f"prices:{symbol}"
    await manager.connect(websocket, topic)
    signal_count = 0
    start_time = time.monotonic()
    logger.info(
        "prices_ws connection opened",
        extra={"topic": topic, "symbol": symbol, "signal_count": signal_count},
    )
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
                signal_count += 1
            except Exception as exc:
                logger.warning(
                    "prices_ws receive error for %s: %s", symbol, exc
                )
                break
    except WebSocketDisconnect:
        pass
    finally:
        duration = time.monotonic() - start_time
        pnl = None  # Placeholder; real P&L should be injected by downstream logic
        logger.info(
            "prices_ws connection closed",
            extra={
                "topic": topic,
                "symbol": symbol,
                "signal_count": signal_count,
                "execution_time_seconds": duration,
                "pnl": pnl,
            },
        )
        manager.disconnect(websocket, topic)