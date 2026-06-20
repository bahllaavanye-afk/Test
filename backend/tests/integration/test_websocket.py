"""WebSocket connection manager tests."""
import json

import pytest

from app.ws.manager import ConnectionManager


class _FakeWebSocket:
    """Minimal stand-in for a Starlette WebSocket that records sent frames.

    ``connect()`` awaits ``accept()`` and ``broadcast()`` awaits ``send_text()``,
    so we implement just those. ``fail`` makes ``send_text`` raise to exercise the
    dead-connection pruning path.
    """

    def __init__(self, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.accepted = False
        self.fail = fail

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, message: str) -> None:
        if self.fail:
            raise RuntimeError("socket closed")
        self.sent.append(message)


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


@pytest.mark.asyncio
async def test_broadcast_delivers_to_subscriber():
    """A connected subscriber must actually receive the broadcast payload.

    This closes the HANDOFF §3 gap where only the no-subscriber path was covered:
    here we assert the exact JSON frame is delivered on the subscribed topic and
    NOT on an unrelated one.
    """
    mgr = ConnectionManager()
    ws = _FakeWebSocket()
    await mgr.connect(ws, "prices:SPY")

    payload = {"type": "quote", "symbol": "SPY", "last": 521.34}
    await mgr.broadcast("prices:SPY", payload)
    # Different topic — must not reach this subscriber.
    await mgr.broadcast("prices:AAPL", {"type": "quote", "symbol": "AAPL"})

    assert ws.accepted is True
    assert len(ws.sent) == 1
    assert json.loads(ws.sent[0]) == payload


@pytest.mark.asyncio
async def test_broadcast_prunes_dead_connection():
    """A connection whose send raises is dropped and doesn't break delivery to others."""
    mgr = ConnectionManager()
    good = _FakeWebSocket()
    dead = _FakeWebSocket(fail=True)
    await mgr.connect(good, "prices:SPY")
    await mgr.connect(dead, "prices:SPY")

    await mgr.broadcast("prices:SPY", {"last": 1.0})

    assert len(good.sent) == 1
    # The dead socket was discarded, so a second broadcast only reaches the good one.
    await mgr.broadcast("prices:SPY", {"last": 2.0})
    assert len(good.sent) == 2
    assert dead not in mgr._connections.get("prices:SPY", set())
