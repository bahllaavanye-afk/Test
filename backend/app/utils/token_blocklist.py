"""
Token revocation utilities using a JTI blocklist.

This module provides helper functions to mark a JWT identifier (JTI) as
revoked and to query the revocation status.  Revoked JTIs are stored in
Redis (via Upstash) when a ``REDIS_URL`` is configured; otherwise an
in‑memory dictionary is used as a fallback, which is suitable for testing
or environments without Redis.  The in‑memory store is cleared on process
restart.

The Redis keys are of the form ``revoked_jti:<jti>`` and are set with a
TTL that matches the remaining lifetime of the token, ensuring automatic
expiration.
"""

from __future__ import annotations

import time
from typing import Any, Optional

# In‑memory fallback: maps JTI to the Unix timestamp at which the entry expires.
_memory_blocklist: dict[str, float] = {}


async def _try_get_redis() -> Optional[Any]:
    """Attempt to obtain a Redis client.

    Returns:
        An ``aioredis.Redis`` instance if ``REDIS_URL`` is configured and the
        connection succeeds, otherwise ``None``.
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
    """Mark a JTI as revoked.

    The JTI is stored with a TTL equal to the remaining lifetime of the token.
    If Redis is available, the revocation is persisted there; otherwise the
    entry is added to the in‑memory blocklist.

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
    """Check whether a JTI has been revoked.

    The function first queries Redis (if configured).  If Redis is unavailable,
    it falls back to the in‑memory blocklist and cleans up any expired entries.

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