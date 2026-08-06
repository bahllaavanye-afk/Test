"""
Utility for handling token revocation using a JTI blocklist.

The blocklist can be stored in Redis (via Upstash) when the ``REDIS_URL`` setting
is defined. If Redis is unavailable, an in‑memory dictionary is used as a fallback,
which is suitable for testing environments. Entries are stored with a TTL that
matches the remaining lifetime of the token, ensuring automatic expiration.
"""

from __future__ import annotations

import time
from typing import Any, Optional

# In‑memory fallback: maps JTI to its expiration timestamp (Unix epoch seconds).
_memory_blocklist: dict[str, float] = {}


async def _try_get_redis() -> Optional[Any]:
    """
    Attempt to obtain a Redis client.

    Returns:
        An ``aioredis`` client instance if ``settings.redis_url`` is configured and
        a connection can be established; otherwise ``None``.
    """
    try:
        from app.config import settings
        if not settings.redis_url:
            return None
        from app.redis_client import get_redis
        r = get_redis()
        await r.ping()
        return r
    except Exception:
        return None


async def revoke_jti(jti: str, ttl_seconds: int) -> None:
    """
    Mark a JWT identifier (JTI) as revoked.

    The JTI is stored with a TTL equal to the token's remaining lifetime.
    If Redis is available the revocation is persisted there; otherwise the
    in‑memory fallback is used.

    Args:
        jti: The JWT identifier to revoke.
        ttl_seconds: Time‑to‑live for the revocation entry, in seconds.
    """
    r = await _try_get_redis()
    if r is not None:
        try:
            await r.setex(f"revoked_jti:{jti}", ttl_seconds, "1")
            return
        except Exception:
            pass
    # Fallback to in‑memory storage
    _memory_blocklist[jti] = time.time() + ttl_seconds


async def is_revoked(jti: str) -> bool:
    """
    Check whether a given JTI has been revoked.

    The function first queries Redis; if unavailable, it falls back to the
    in‑memory blocklist. Expired entries are cleaned up automatically.

    Args:
        jti: The JWT identifier to check.

    Returns:
        ``True`` if the JTI is present in the blocklist, otherwise ``False``.
    """
    r = await _try_get_redis()
    if r is not None:
        try:
            return bool(await r.exists(f"revoked_jti:{jti}"))
        except Exception:
            pass
    # Fallback to in‑memory blocklist
    expires = _memory_blocklist.get(jti)
    if expires is None:
        return False
    if time.time() > expires:
        _memory_blocklist.pop(jti, None)
        return False
    return True