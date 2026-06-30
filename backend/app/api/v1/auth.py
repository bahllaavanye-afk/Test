from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.utils.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.utils.token_blocklist import revoke_jti, is_revoked
from app.utils.exceptions import UnauthorizedError

# ── Constants ───────────────────────────────────────────────────────────────
DEFAULT_ENDPOINT = "login"
DEFAULT_LIMIT = 20
LOGIN_LIMIT = 20
REGISTER_LIMIT = 10
REFRESH_LIMIT = 30
WINDOW_SECONDS = 60
RETRY_AFTER_SECONDS = "60"

EMAIL_ALREADY_REGISTERED_STATUS = 409
EMAIL_ALREADY_REGISTERED_DETAIL = "Email already registered"

DEMO_MODE_DISABLED_STATUS = 404
DEMO_MODE_DISABLED_DETAIL = "Demo mode is disabled"
DEMO_EMAIL = "demo@quantedge.app"

GOOGLE_OAUTH_NOT_CONFIGURED_STATUS = 503
GOOGLE_OAUTH_NOT_CONFIGURED_DETAIL = "Google OAuth not configured"

INVALID_REFRESH_TOKEN_DETAIL = "Invalid refresh token"
INVALID_LOGIN_DETAIL = "Invalid email or password"

DEFAULT_BEARER_TOKEN_TYPE = "bearer"

DEFAULT_PAPER_BROKER = "alpaca"
DEFAULT_PAPER_MODE = "paper"
DEFAULT_PAPER_LABEL = "Paper Account"
DEFAULT_PAPER_EQUITY = 100_000.0
DEFAULT_PAPER_CASH = 100_000.0

USER_AGENT_MAX_LENGTH = 256

# ── Router ───────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/auth", tags=["auth"])

# Simple in-memory brute-force protection (per IP, resets on restart)
# Production: use Redis-backed counter with TTL via Upstash
_login_attempts: dict[str, list[float]] = defaultdict(list)

# Per-endpoint limits (attempts per 60-second sliding window)
_ENDPOINT_LIMITS: dict[str, int] = {
    "login": LOGIN_LIMIT,
    "register": REGISTER_LIMIT,
    "refresh": REFRESH_LIMIT,
}
_WINDOW_SECONDS = WINDOW_SECONDS  # alias for readability


def _check_rate_limit(ip: str, endpoint: str = DEFAULT_ENDPOINT) -> None:
    import time

    now = time.time()
    window_start = now - _WINDOW_SECONDS
    key = f"{ip}:{endpoint}"
    attempts = [t for t in _login_attempts[key] if t > window_start]
    _login_attempts[key] = attempts
    max_attempts = _ENDPOINT_LIMITS.get(endpoint, DEFAULT_LIMIT)
    if len(attempts) >= max_attempts:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Limit is {max_attempts}/minute.",
            headers={"Retry-After": RETRY_AFTER_SECONDS},
        )
    _login_attempts[key].append(now)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = DEFAULT_BEARER_TOKEN_TYPE


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)):
    # Same brute-force protection as /login — prevents enumeration and abuse
    _check_rate_limit(request.client.host if request.client else "unknown", "register")
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=EMAIL_ALREADY_REGISTERED_STATUS, detail=EMAIL_ALREADY_REGISTERED_DETAIL)

    user = User(
        id=str(uuid.uuid4()),
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Auto-create paper account for new user
    from app.models.account import Account

    paper_account = Account(
        user_id=user.id,
        broker=DEFAULT_PAPER_BROKER,
        mode=DEFAULT_PAPER_MODE,
        label=DEFAULT_PAPER_LABEL,
        extra_config={"equity": DEFAULT_PAPER_EQUITY, "cash": DEFAULT_PAPER_CASH},
    )
    db.add(paper_account)
    await db.commit()

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
        raise UnauthorizedError(INVALID_LOGIN_DETAIL)

    # Audit log for successful login
    from app.models.audit_log import AuditLog

    log = AuditLog(
        user_id=user.id,
        action="login",
        resource_type="user",
        resource_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent", "")[:USER_AGENT_MAX_LENGTH],
        extra_data={},
    )
    db.add(log)
    await db.commit()

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/demo", response_model=TokenResponse)
async def demo_login(db: AsyncSession = Depends(get_db)):
    """Issue a token for a shared demo user so the login-free public app is functional
    (every page/button needs a JWT). Gated by DEMO_MODE — disable for real multi-user.
    """
    if not settings.demo_mode:
        raise HTTPException(status_code=DEMO_MODE_DISABLED_STATUS, detail=DEMO_MODE_DISABLED_DETAIL)
    import secrets as _secrets

    email = DEMO_EMAIL
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            hashed_password=hash_password(_secrets.token_urlsafe(32)),
            is_active=True,
            is_superuser=False,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
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
        raise UnauthorizedError(INVALID_REFRESH_TOKEN_DETAIL)

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


# ── Google OAuth ──────────────────────────────────────────────────────────────

@router.get("/google")
async def google_oauth_start():
    """Redirect to Google OAuth consent screen."""
    if not settings.google_client_id:
        raise HTTPException(status_code=GOOGLE_OAUTH_NOT_CONFIGURED_STATUS, detail=GOOGLE_OAUTH_NOT_CONFIGURED_DETAIL)

    import urllib.parse

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return RedirectResponse(url)