"""
Refresh-token revocation via JTI blocklist.

Uses Redis (Upstash) when REDIS_URL is set; falls back to an in‑memory
set otherwise (good for testing, resets on restart).

Stored keys: "revoked_jti:<jti>" with TTL = token remaining lifetime.
"""
from __future__ import annotations

import logging
import time
from typing import Final

# In-memory fallback: {jti: expires_at_unix}
_memory_blocklist: dict[str, float] = {}

# Structured logger – assumes the application configures a JSON/structured logger.
_logger: Final = logging.getLogger(__name__)

# Simple metrics – kept in‑process; in production these could be exported to Prometheus, etc.
_revoked_counter: int = 0
_check_counter: int = 0


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


async def revoke_jti(jti: str, ttl_seconds: int) -> None:
    """Mark this JTI as revoked. TTL should equal the token's remaining lifetime."""
    global _revoked_counter
    start = time.monotonic()
    r = await _try_get_redis()
    if r is not None:
        try:
            await r.setex(f"revoked_jti:{jti}", ttl_seconds, "1")
            _revoked_counter += 1
            _logger.info(
                "revoke_jti.success",
                jti=jti,
                ttl_seconds=ttl_seconds,
                execution_ms=int((time.monotonic() - start) * 1000),
                total_revoked=_revoked_counter,
                pnl=0,  # placeholder for P&L metric
            )
            return
        except Exception as exc:
            _logger.info(
                "revoke_jti.redis_error",
                jti=jti,
                ttl_seconds=ttl_seconds,
                error=str(exc),
                execution_ms=int((time.monotonic() - start) * 1000),
                total_revoked=_revoked_counter,
                pnl=0,
            )
    # Fallback to in-memory
    _memory_blocklist[jti] = time.time() + ttl_seconds
    _revoked_counter += 1
    _logger.info(
        "revoke_jti.memory_fallback",
        jti=jti,
        ttl_seconds=ttl_seconds,
        execution_ms=int((time.monotonic() - start) * 1000),
        total_revoked=_revoked_counter,
        pnl=0,
    )


async def is_revoked(jti: str) -> bool:
    """Return True if this JTI has been revoked."""
    global _check_counter
    start = time.monotonic()
    _check_counter += 1
    r = await _try_get_redis()
    if r is not None:
        try:
            exists = bool(await r.exists(f"revoked_jti:{jti}"))
            _logger.info(
                "is_revoked.redis_check",
                jti=jti,
                revoked=exists,
                execution_ms=int((time.monotonic() - start) * 1000),
                check_count=_check_counter,
                pnl=0,
            )
            return exists
        except Exception as exc:
            _logger.info(
                "is_revoked.redis_error",
                jti=jti,
                error=str(exc),
                execution_ms=int((time.monotonic() - start) * 1000),
                check_count=_check_counter,
                pnl=0,
            )
    # Fallback to in-memory
    expires = _memory_blocklist.get(jti)
    if expires is None:
        _logger.info(
            "is_revoked.memory_miss",
            jti=jti,
            revoked=False,
            execution_ms=int((time.monotonic() - start) * 1000),
            check_count=_check_counter,
            pnl=0,
        )
        return False
    if time.time() > expires:
        _memory_blocklist.pop(jti, None)
        _logger.info(
            "is_revoked.memory_expired",
            jti=jti,
            revoked=False,
            execution_ms=int((time.monotonic() - start) * 1000),
            check_count=_check_counter,
            pnl=0,
        )
        return False
    _logger.info(
        "is_revoked.memory_hit",
        jti=jti,
        revoked=True,
        execution_ms=int((time.monotonic() - start) * 1000),
        check_count=_check_counter,
        pnl=0,
    )
    return True