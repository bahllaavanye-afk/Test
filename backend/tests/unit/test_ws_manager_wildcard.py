"""Wildcard fan-out for the WebSocket ConnectionManager.

Regression test for the `/ws/prices` all-symbols bug: the all-symbols socket
subscribes to the literal topic ``prices:*`` while the feed broadcasts to
``prices:{symbol}``. Before the fix, the wildcard subscriber received nothing.
"""
import pytest
from pydantic import BaseModel, Field, validator

from app.ws.manager import ConnectionManager


class PriceMessage(BaseModel):
    """Schema for price update messages broadcast over the ``prices`` channel.

    Attributes
    ----------
    symbol: str
        Ticker symbol for which the price update is emitted. Must be non‑empty.
    last: float
        The latest traded price. Must be a positive number.
    """

    symbol: str = Field(
        ...,
        description="Ticker symbol for which the price update is emitted.",
        example="AAPL",
        min_length=1,
    )
    last: float = Field(
        ...,
        description="The latest traded price. Must be greater than zero.",
        example=123.45,
        gt=0,
    )

    @validator("symbol")
    def strip_whitespace(cls, v: str) -> str:
        """Remove surrounding whitespace from the symbol."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("symbol must contain non‑whitespace characters")
        return cleaned


class AlertMessage(BaseModel):
    """Schema for generic alert messages broadcast over the ``alerts`` channel.

    Attributes
    ----------
    msg: str
        Human‑readable alert description.
    """

    msg: str = Field(
        ...,
        description="Human‑readable alert description.",
        example="VaR breach",
        min_length=1,
    )

    @validator("msg")
    def non_empty(cls, v: str) -> str:
        """Ensure the alert message is not empty or whitespace only."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("msg must contain non‑whitespace characters")
        return cleaned


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