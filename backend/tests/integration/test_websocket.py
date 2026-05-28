"""WebSocket connection manager tests."""
import pytest
from app.ws.manager import ConnectionManager


@pytest.mark.asyncio
async def test_connection_manager_broadcast_no_subscribers():
    mgr = ConnectionManager()
    # Should not raise when no subscribers
    await mgr.broadcast("topic", {"test": "data"})


@pytest.mark.asyncio
async def test_connection_manager_disconnect_unsubscribed():
    mgr = ConnectionManager()
    # Disconnect without connect should not raise
    mgr.disconnect(None, "test_topic")  # type: ignore
