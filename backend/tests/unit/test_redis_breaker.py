"""Unit tests for Redis circuit breaker behavior.

The circuit breaker should trip on connection-related errors, disabling
further Redis operations, while allowing non‑critical operational errors
to pass without disabling the client.
"""

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

import app.redis_client as rc


@pytest.fixture(autouse=True)
def _reset_breaker() -> None:
    """Reset the breaker state before and after each test."""
    rc._redis_tripped = False
    yield
    rc._redis_tripped = False


def test_connection_error_trips_breaker() -> None:
    """A Redis connection error must trip the breaker and disable the client."""
    rc._note_redis_error("get", RedisConnectionError("Connection refused"))
    assert rc._redis_tripped is True
    # The enabled flag respects the tripped state.
    assert rc._redis_enabled() is False


def test_os_error_trips_breaker() -> None:
    """An OS level error (e.g., network down) should also trip the breaker."""
    rc._note_redis_error("set", OSError("network down"))
    assert rc._redis_tripped is True


def test_op_error_does_not_trip_breaker() -> None:
    """Operational errors (e.g., bad JSON) must NOT trip the breaker."""
    rc._note_redis_error("get_price", ValueError("bad payload"))
    assert rc._redis_tripped is False


@pytest.mark.asyncio
async def test_pricecache_noops_after_trip() -> None:
    """After the breaker trips, the PriceCache should become a no‑op."""
    # Bypass __init__ to avoid real connection pool creation.
    pc = rc.PriceCache.__new__(rc.PriceCache)

    class _Boom:
        """Mock Redis client that always raises a connection error."""

        async def setex(self, *args, **kwargs) -> None:
            raise RedisConnectionError("Connection refused")

        async def get(self, *args, **kwargs) -> None:
            raise RedisConnectionError("Connection refused")

    pc._r = _Boom()

    # First operation fails, trips the breaker, but does not raise.
    await pc.set_price("binance", "AAPL", {"last": 1.0})
    assert rc._redis_tripped is True

    # Subsequent calls should short‑circuit to a no‑op.
    assert pc._client() is None
    assert await pc.get_price("binance", "AAPL") is None