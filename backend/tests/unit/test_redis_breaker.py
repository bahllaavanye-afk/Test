"""Redis circuit breaker: a connection failure must trip the breaker once and then
no-op (and stop logging) instead of retrying on every call.
"""
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

import app.redis_client as rc


@pytest.fixture(autouse=True)
def _reset_breaker():
    rc._redis_tripped = False
    yield
    rc._redis_tripped = False


def test_connection_error_trips_breaker():
    rc._note_redis_error("get", RedisConnectionError("Connection refused"))
    assert rc._redis_tripped is True
    assert rc._redis_enabled() is False  # enabled() honours the breaker


def test_os_error_trips_breaker():
    rc._note_redis_error("set", OSError("network down"))
    assert rc._redis_tripped is True


def test_op_error_does_not_trip_breaker():
    # A one‑off operational error (e.g. bad JSON) must NOT disable Redis.
    rc._note_redis_error("get_price", ValueError("bad payload"))
    assert rc._redis_tripped is False


def test_non_fatal_then_fatal_errors():
    # First a non‑fatal error should not trip the breaker.
    rc._note_redis_error("fetch", ValueError("malformed data"))
    assert rc._redis_tripped is False

    # A subsequent fatal connection error must trip the breaker.
    rc._note_redis_error("fetch", RedisConnectionError("Connection refused"))
    assert rc._redis_tripped is True
    assert rc._redis_enabled() is False


@pytest.mark.asyncio
async def test_pricecache_noops_after_trip():
    pc = rc.PriceCache.__new__(rc.PriceCache)  # bypass __init__ (no real pool)

    class _Boom:
        async def setex(self, *a, **k):
            raise RedisConnectionError("Connection refused")

        async def get(self, *a, **k):
            raise RedisConnectionError("Connection refused")

    pc._r = _Boom()

    # First op hits Redis, fails, trips the breaker — but never raises.
    await pc.set_price("binance", "AAPL", {"last": 1.0})
    assert rc._redis_tripped is True

    # After tripping, the client is gone and ops short‑circuit to no‑op.
    assert pc._client() is None
    assert await pc.get_price("binance", "AAPL") is None


@pytest.mark.asyncio
async def test_pricecache_noops_without_error():
    pc = rc.PriceCache.__new__(rc.PriceCache)  # bypass __init__ (no real pool)

    class _Silent:
        async def setex(self, *a, **k):
            return True

        async def get(self, *a, **k):
            return b'{"last":2.0}'

    pc._r = _Silent()

    # Normal operation should succeed and not trip the breaker.
    await pc.set_price("binance", "MSFT", {"last": 2.0})
    assert rc._redis_tripped is False
    price = await pc.get_price("binance", "MSFT")
    assert price is not None
    # The client remains available after successful calls.
    assert pc._client() is not None