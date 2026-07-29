"""
Refresh-token revocation via JTI blocklist.

Uses Redis (Upstash) when REDIS_URL is set; falls back to an in-memory
set otherwise (good for testing, resets on restart).

Stored keys: "revoked_jti:<jti>" with TTL = token remaining lifetime.
"""
from __future__ import annotations

import time
import logging
from typing import Optional

# In-memory fallback: {jti: expires_at_unix}
_memory_blocklist: dict[str, float] = {}

_logger = logging.getLogger(__name__)

async def _try_get_redis() -> Optional["aioredis.Redis"]:
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


async def revoke_jti(jti: str, ttl_seconds: int) -> None:
    """Mark this JTI as revoked. TTL should equal the token's remaining lifetime."""
    start = time.perf_counter()
    r = await _try_get_redis()
    if r is not None:
        try:
            await r.setex(f"revoked_jti:{jti}", ttl_seconds, "1")
            duration_ms = (time.perf_counter() - start) * 1000
            _logger.info(
                "revoke_jti_success",
                extra={
                    "jti": jti,
                    "ttl_seconds": ttl_seconds,
                    "backend": "redis",
                    "duration_ms": round(duration_ms, 2),
                    "blocklist_size": len(_memory_blocklist),
                },
            )
            return
        except Exception:
            pass
    # Fallback to in-memory
    _memory_blocklist[jti] = time.time() + ttl_seconds
    duration_ms = (time.perf_counter() - start) * 1000
    _logger.info(
        "revoke_jti_fallback",
        extra={
            "jti": jti,
            "ttl_seconds": ttl_seconds,
            "backend": "memory",
            "duration_ms": round(duration_ms, 2),
            "blocklist_size": len(_memory_blocklist),
        },
    )


async def is_revoked(jti: str) -> bool:
    """Return True if this JTI has been revoked."""
    start = time.perf_counter()
    r = await _try_get_redis()
    if r is not None:
        try:
            result = bool(await r.exists(f"revoked_jti:{jti}"))
            duration_ms = (time.perf_counter() - start) * 1000
            _logger.info(
                "is_revoked_redis",
                extra={
                    "jti": jti,
                    "revoked": result,
                    "backend": "redis",
                    "duration_ms": round(duration_ms, 2),
                    "blocklist_size": len(_memory_blocklist),
                },
            )
            return result
        except Exception:
            pass
    # Fallback to in-memory
    expires = _memory_blocklist.get(jti)
    if expires is None:
        result = False
    else:
        if time.time() > expires:
            _memory_blocklist.pop(jti, None)
            result = False
        else:
            result = True
    duration_ms = (time.perf_counter() - start) * 1000
    _logger.info(
        "is_revoked_memory",
        extra={
            "jti": jti,
            "revoked": result,
            "backend": "memory",
            "duration_ms": round(duration_ms, 2),
            "blocklist_size": len(_memory_blocklist),
        },
    )
    return result