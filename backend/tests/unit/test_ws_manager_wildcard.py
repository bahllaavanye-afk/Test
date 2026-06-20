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


@pytest.mark.asyncio
async def test_wildcard_subscriber_receives_concrete_topic():
    m = ConnectionManager()
    all_sub, one_sub = _FakeWS(), _FakeWS()
    await m.connect(all_sub, "prices:*")
    await m.connect(one_sub, "prices:AAPL")

    await m.broadcast("prices:AAPL", {"symbol": "AAPL", "last": 1.0})
    assert len(all_sub.sent) == 1, "wildcard subscriber must receive prices:AAPL"
    assert len(one_sub.sent) == 1, "exact-topic subscriber must still receive its symbol"

    # A different symbol reaches the wildcard, not the AAPL-only socket.
    await m.broadcast("prices:TSLA", {"symbol": "TSLA", "last": 2.0})
    assert len(all_sub.sent) == 2
    assert len(one_sub.sent) == 1


@pytest.mark.asyncio
async def test_wildcard_is_prefix_scoped():
    """A ``prices:*`` subscriber must NOT receive a different prefix's broadcasts."""
    m = ConnectionManager()
    price_sub = _FakeWS()
    await m.connect(price_sub, "prices:*")
    await m.broadcast("alerts:risk", {"msg": "VaR breach"})
    assert price_sub.sent == []


@pytest.mark.asyncio
async def test_dead_socket_is_purged():
    class _Dead(_FakeWS):
        async def send_text(self, message: str) -> None:
            raise RuntimeError("connection closed")

    m = ConnectionManager()
    dead = _Dead()
    await m.connect(dead, "prices:*")
    await m.broadcast("prices:AAPL", {"symbol": "AAPL"})
    # Purged from every topic set so it is not retried forever.
    assert all(dead not in s for s in m._connections.values())
