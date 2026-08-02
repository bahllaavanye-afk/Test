"""Real-time order status WebSocket endpoint."""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/orders")
async def orders_ws(websocket: WebSocket):
    topic = "orders"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                await websocket.receive_text()
            except Exception as exc:
                logger.warning("orders_ws receive error: %s", exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, topic)


# ---------------------------------------------------------------------------
# Unit tests for edge cases
# ---------------------------------------------------------------------------
import pytest
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def app():
    """Create a FastAPI app with the orders router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_immediate_disconnect(app):
    """
    Edge case: client connects and then disconnects before sending any data.
    Verify that manager.disconnect is called and no unhandled exceptions occur.
    """
    # Patch manager methods to track calls
    with patch("app.ws.manager.manager") as mock_manager:
        mock_manager.connect = AsyncMock()
        mock_manager.disconnect = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/ws/orders") as websocket:
            # Immediately close the connection
            websocket.close()

        # Ensure connect was called once and disconnect was called once
        mock_manager.connect.assert_awaited_once()
        mock_manager.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_text_raises_exception(app):
    """
    Edge case: websocket.receive_text raises an unexpected exception.
    The endpoint should log the warning, break the loop, and call disconnect.
    """
    # Create a mock WebSocket that raises on receive_text
    class MockWebSocket(WebSocket):
        async def receive_text(self):
            raise RuntimeError("simulated receive failure")

    mock_ws = MockWebSocket(scope={"type": "websocket"}, receive=MagicMock(), send=MagicMock())

    with patch("app.ws.manager.manager") as mock_manager:
        mock_manager.connect = AsyncMock()
        mock_manager.disconnect = AsyncMock()

        # Directly invoke the endpoint coroutine
        with pytest.raises(RuntimeError):
            # The exception propagates out of receive_text, but the endpoint catches it.
            # We run the coroutine manually to observe behavior.
            task = asyncio.create_task(orders_ws(mock_ws))
            # Allow the coroutine to run a short while
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Verify that connect and disconnect were still awaited
        mock_manager.connect.assert_awaited_once()
        mock_manager.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_multiple_receive_cycles_until_disconnect(app):
    """
    Edge case: client sends several valid messages before disconnecting.
    Verify that the loop continues correctly and disconnect is called after the client disconnects.
    """
    with patch("app.ws.manager.manager") as mock_manager:
        mock_manager.connect = AsyncMock()
        mock_manager.disconnect = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/ws/orders") as websocket:
            # Send a few messages
            for _ in range(3):
                websocket.send_text("ping")
            # Close the connection to trigger WebSocketDisconnect handling
            websocket.close()

        mock_manager.connect.assert_awaited_once()
        mock_manager.disconnect.assert_awaited_once()