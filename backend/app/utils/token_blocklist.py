"""
Utility for managing a blocklist of revoked JWT IDs (JTIs).

The blocklist can be stored in Redis (via Upstash) when a Redis URL is
configured, or fall back to an in‑memory dictionary for testing or when
Redis is unavailable. Each revoked JTI is stored with a TTL that matches
the remaining lifetime of the associated token, ensuring automatic
expiration.

Functions
---------
- `revoke_jti(jti: str, ttl_seconds: int) -> None`
    Mark a JTI as revoked for the given TTL.
- `is_revoked(jti: str) -> bool`
    Check whether a JTI is currently revoked.
"""

from __future__ import annotations

import time
from typing import Optional, Any

# In‑memory fallback storage: maps JTI to its expiration timestamp (Unix time).
_memory_blocklist: dict[str, float] = {}


async def _try_get_redis() -> Optional[Any]:
    """
    Attempt to obtain a Redis client.

    Returns
    -------
    Optional[Any]
        An `aioredis.Redis` instance if `REDIS_URL` is configured and a
        connection can be established; otherwise ``None``.
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
    Mark the specified JTI as revoked.

    The JTI is stored with a TTL equal to the remaining lifetime of the
    token, after which it will be automatically removed from the blocklist.

    Parameters
    ----------
    jti : str
        The JWT ID to revoke.
    ttl_seconds : int
        Time‑to‑live for the revocation entry, in seconds.
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
    Determine whether a JTI has been revoked.

    Checks Redis first (if available); on failure or absence of Redis,
    falls back to the in‑memory blocklist. Expired entries in the
    in‑memory store are cleaned up automatically.

    Parameters
    ----------
    jti : str
        The JWT ID to query.

    Returns
    -------
    bool
        ``True`` if the JTI is currently revoked; ``False`` otherwise.
    """
    r = await _try_get_redis()
    if r is not None:
        try:
            return bool(await r.exists(f"revoked_jti:{jti}"))
        except Exception:
            pass
    # Fallback to in‑memory storage
    expires = _memory_blocklist.get(jti)
    if expires is None:
        return False
    if time.time() > expires:
        _memory_blocklist.pop(jti, None)
        return False
    return True