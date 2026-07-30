"""
Refresh-token revocation via JTI blocklist.

Uses Redis (Upstash) when REDIS_URL is set; falls back to an in-memory
set otherwise (good for testing, resets on restart).

Stored keys: "revoked_jti:<jti>" with TTL = token remaining lifetime.
"""
from __future__ import annotations

import time
import asyncio
from typing import Dict

# In-memory fallback: {jti: expires_at_unix}
_memory_blocklist: Dict[str, float] = {}
_memory_lock = asyncio.Lock()


async def _try_get_redis():
    """Return an aioredis.Redis instance if REDIS_URL is configured and reachable, else None."""
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


async def _cleanup_memory_blocklist() -> None:
    """Remove expired JTIs from the in‑memory blocklist."""
    now = time.time()
    async with _memory_lock:
        expired = [jti for jti, expires in _memory_blocklist.items() if expires <= now]
        for jti in expired:
            _memory_blocklist.pop(jti, None)


async def revoke_jti(jti: str, ttl_seconds: int) -> None:
    """
    Mark the given JTI as revoked.

    The TTL should match the token's remaining lifetime. If Redis is
    available the revocation is stored there; otherwise it falls back
    to an in‑memory dictionary.
    """
    if ttl_seconds <= 0:
        # No need to store a revocation that would expire immediately.
        return

    r = await _try_get_redis()
    if r is not None:
        try:
            await r.setex(f"revoked_jti:{jti}", ttl_seconds, "1")
            return
        except Exception:
            # Redis operation failed; fall back to in‑memory.
            pass

    expires_at = time.time() + ttl_seconds
    async with _memory_lock:
        _memory_blocklist[jti] = expires_at


async def is_revoked(jti: str) -> bool:
    """
    Check whether a JTI has been revoked.

    Returns True if the JTI is present in Redis or the in‑memory blocklist.
    Expired entries are cleaned up automatically.
    """
    r = await _try_get_redis()
    if r is not None:
        try:
            return bool(await r.exists(f"revoked_jti:{jti}"))
        except Exception:
            # Redis check failed; fall back to in‑memory.
            pass

    # In‑memory check with cleanup of stale entries.
    await _cleanup_memory_blocklist()
    async with _memory_lock:
        expires = _memory_blocklist.get(jti)
        if expires is None:
            return False
        if time.time() > expires:
            # Expired; remove it.
            _memory_blocklist.pop(jti, None)
            return False
        return True