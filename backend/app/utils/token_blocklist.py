"""
Refresh-token revocation via JTI blocklist.

Uses Redis (Upstash) when REDIS_URL is set; falls back to an in‑memory
set otherwise (good for testing, resets on restart).

Stored keys: "revoked_jti:<jti>" with TTL = token remaining lifetime.
"""
from __future__ import annotations

import time

# In‑memory fallback: {jti: expires_at_unix}
_memory_blocklist: dict[str, float] = {}


async def _try_get_redis():
    """Return an aioredis.Redis instance if REDIS_URL is configured and reachable; otherwise None."""
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
    """Mark the given JTI as revoked. TTL should equal the token's remaining lifetime."""
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
    """Return True if the JTI is revoked, otherwise False."""
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