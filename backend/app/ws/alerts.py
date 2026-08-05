"""Strategy signal alerts WebSocket endpoint."""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket):
    topic = "alerts"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                await websocket.receive_text()
            except Exception as exc:  # pragma: no cover
                logger.warning("alerts_ws receive error: %s", exc)
                break
    except WebSocketDisconnect:  # pragma: no cover
        pass
    finally:
        await manager.disconnect(websocket, topic)


# ---------------------------------------------------------------------------
# Unit tests for edge cases
# ---------------------------------------------------------------------------
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def app():
    """FastAPI app with the alerts router attached."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_alerts_ws_connect_and_disconnect(client):
    """Ensure manager.connect and manager.disconnect are called exactly once."""
    async_connect = AsyncMock()
    async_disconnect = AsyncMock()
    with patch.object(manager, "connect", async_connect), patch.object(
        manager, "disconnect", async_disconnect
    ):
        with client.websocket_connect("/ws/alerts") as websocket:
            # Immediately close the connection; the server should break out of the loop
            websocket.close()
        # After the context exits, the server should have called connect and disconnect
        async_connect.assert_awaited_once()
        async_disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_alerts_ws_receive_error_handling(client):
    """Simulate a receive error after a normal message and verify graceful shutdown."""
    async_connect = AsyncMock()
    async_disconnect = AsyncMock()
    with patch.object(manager, "connect", async_connect), patch.object(
        manager, "disconnect", async_disconnect
    ):
        with client.websocket_connect("/ws/alerts") as websocket:
            # Send a normal message; the server will receive it without issue
            websocket.send_text("test")
            # Force a protocol error by closing the socket abruptly
            websocket.close()
        async_connect.assert_awaited_once()
        async_disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_alerts_ws_large_payload_handling(client):
    """Send a large payload (1 MB) to ensure the endpoint does not crash."""
    async_connect = AsyncMock()
    async_disconnect = AsyncMock()
    large_message = "x" * 1_048_576  # 1 MB of data
    with patch.object(manager, "connect", async_connect), patch.object(
        manager, "disconnect", async_disconnect
    ):
        with client.websocket_connect("/ws/alerts") as websocket:
            websocket.send_text(large_message)
            websocket.close()
        async_connect.assert_awaited_once()
        async_disconnect.assert_awaited_once()