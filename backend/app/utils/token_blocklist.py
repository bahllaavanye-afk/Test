"""
Refresh-token revocation via JTI blocklist.

Uses Redis (Upstash) when REDIS_URL is set; falls back to an in-memory
set otherwise (good for testing, resets on restart).

Stored keys: "revoked_jti:<jti>" with TTL = token remaining lifetime.
"""
from __future__ import annotations

import time
from typing import Optional

# In-memory fallback: {jti: expires_at_unix}
_memory_blocklist: dict[str, float] = {}


async def _try_get_redis():
    """Return aioredis.Redis if REDIS_URL is configured and reachable, else None."""
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


async def revoke_jti(jti: Optional[str], ttl_seconds: Optional[int]) -> None:
    """
    Mark this JTI as revoked. TTL should equal the token's remaining lifetime.

    Edge‑case handling:
    * If `jti` is None or an empty string, the function returns early.
    * If `ttl_seconds` is missing, non‑positive, or not an int, the JTI is not stored.
    """
    if not jti:
        # Nothing to revoke
        return

    if not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
        # Non‑positive TTL means the token is already expired; no need to store.
        return

    r = await _try_get_redis()
    if r is not None:
        try:
            await r.setex(f"revoked_jti:{jti}", ttl_seconds, "1")
            return
        except Exception:
            pass
    # Fallback to in-memory
    _memory_blocklist[jti] = time.time() + ttl_seconds


async def is_revoked(jti: Optional[str]) -> bool:
    """
    Return True if this JTI has been revoked.

    Edge‑case handling:
    * Returns False for None or empty JTI.
    * Cleans up expired entries that hit the exact expiration moment.
    """
    if not jti:
        return False

    r = await _try_get_redis()
    if r is not None:
        try:
            return bool(await r.exists(f"revoked_jti:{jti}"))
        except Exception:
            pass
    # Fallback to in-memory
    expires = _memory_blocklist.get(jti)
    if expires is None:
        return False
    # Consider the token revoked if the current time is greater than or equal to expiration.
    if time.time() >= expires:
        _memory_blocklist.pop(jti, None)
        return False
    return True