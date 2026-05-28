from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.utils.security import (
    create_access_token, create_refresh_token, decode_token,
    hash_password, verify_password,
)
from app.utils.token_blocklist import revoke_jti, is_revoked
from app.utils.exceptions import UnauthorizedError

router = APIRouter(prefix="/auth", tags=["auth"])

# Simple in-memory brute-force protection (per IP, resets on restart)
# Production: use Redis-backed counter with TTL via Upstash
_login_attempts: dict[str, list[float]] = defaultdict(list)
_MAX_ATTEMPTS = 10
_WINDOW_SECONDS = 300  # 10 attempts per 5 minutes per IP


def _check_rate_limit(ip: str) -> None:
    import time
    now = time.time()
    window_start = now - _WINDOW_SECONDS
    attempts = [t for t in _login_attempts[ip] if t > window_start]
    _login_attempts[ip] = attempts
    if len(attempts) >= _MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Try again in 5 minutes.",
            headers={"Retry-After": "300"},
        )
    _login_attempts[ip].append(now)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)):
    # Same brute-force protection as /login — prevents enumeration and abuse
    _check_rate_limit(request.client.host if request.client else "unknown")
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        id=str(uuid.uuid4()),
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    _check_rate_limit(request.client.host if request.client else "unknown")
    result = await db.execute(select(User).where(User.email == body.email, User.is_active == True))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise UnauthorizedError("Invalid email or password")

    # Audit log for successful login
    from app.models.audit_log import AuditLog
    log = AuditLog(
        user_id=user.id,
        action="login",
        resource_type="user",
        resource_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent", "")[:256],
        extra_data={},
    )
    db.add(log)
    await db.commit()

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            raise UnauthorizedError()
        user_id = payload.get("sub")
        jti = payload.get("jti")
    except Exception:
        raise UnauthorizedError("Invalid refresh token")

    # Reject revoked tokens (logout / rotation)
    if jti and await is_revoked(jti):
        raise UnauthorizedError("Refresh token has been revoked")

    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    user = result.scalar_one_or_none()
    if not user:
        raise UnauthorizedError()

    # Rotate: revoke the consumed token before issuing a new pair
    if jti:
        import time
        exp = payload.get("exp", 0)
        ttl = max(1, int(exp - time.time()))
        await revoke_jti(jti, ttl)

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


class LogoutRequest(BaseModel):
    refresh_token: str


@router.post("/logout", status_code=204)
async def logout(body: LogoutRequest):
    """Revoke the given refresh token. The client must discard the access token too."""
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            return  # silently accept invalid type — token is already useless
        jti = payload.get("jti")
        if jti:
            import time
            exp = payload.get("exp", 0)
            ttl = max(1, int(exp - time.time()))
            await revoke_jti(jti, ttl)
    except Exception:
        pass  # expired/malformed tokens are already invalid — no need to raise
