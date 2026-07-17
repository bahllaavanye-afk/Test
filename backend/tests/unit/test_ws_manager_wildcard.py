"""Wildcard fan-out for the WebSocket ConnectionManager.

Regression test for the `/ws/prices` all-symbols bug: the all-symbols socket
subscribes to the literal topic ``prices:*`` while the feed broadcasts to
``prices:{symbol}``. Before the fix, the wildcard subscriber received nothing.
"""
import pytest

from app.ws.manager import ConnectionManager


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, message: str) -> None:
        self.sent.append(message)


@pytest.fixture
async def manager() -> ConnectionManager:
    """Provide a fresh ConnectionManager for each test."""
    return ConnectionManager()


@pytest.fixture
def fake_ws() -> _FakeWS:
    """Factory for a fresh _FakeWS instance."""
    return _FakeWS()


@pytest.fixture
def dead_ws() -> _FakeWS:
    """Factory for a dead WebSocket that raises on send."""
    class _Dead(_FakeWS):
        async def send_text(self, message: str) -> None:
            raise RuntimeError("connection closed")
    return _Dead()


@pytest.mark.asyncio
async def test_wildcard_subscriber_receives_concrete_topic(manager: ConnectionManager, fake_ws):
    all_sub = fake_ws()
    one_sub = fake_ws()
    await manager.connect(all_sub, "prices:*")
    await manager.connect(one_sub, "prices:AAPL")

    await manager.broadcast("prices:AAPL", {"symbol": "AAPL", "last": 1.0})
    assert len(all_sub.sent) == 1, "wildcard subscriber must receive prices:AAPL"
    assert len(one_sub.sent) == 1, "exact-topic subscriber must still receive its symbol"

    # A different symbol reaches the wildcard, not the AAPL-only socket.
    await manager.broadcast("prices:TSLA", {"symbol": "TSLA", "last": 2.0})
    assert len(all_sub.sent) == 2
    assert len(one_sub.sent) == 1


@pytest.mark.asyncio
async def test_wildcard_is_prefix_scoped(manager: ConnectionManager, fake_ws):
    """A ``prices:*`` subscriber must NOT receive a different prefix's broadcasts."""
    price_sub = fake_ws()
    await manager.connect(price_sub, "prices:*")
    await manager.broadcast("alerts:risk", {"msg": "VaR breach"})
    assert price_sub.sent == []


@pytest.mark.asyncio
async def test_dead_socket_is_purged(manager: ConnectionManager, dead_ws):
    dead = dead_ws()
    await manager.connect(dead, "prices:*")
    await manager.broadcast("prices:AAPL", {"symbol": "AAPL"})
    # Purged from every topic set so it is not retried forever.
    assert all(dead not in s for s in manager._connections.values())