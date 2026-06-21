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
    create_access_token, create_refresh_token, decode_token,
    hash_password, verify_password,
)
from app.utils.token_blocklist import revoke_jti, is_revoked
from app.utils.exceptions import UnauthorizedError

router = APIRouter(prefix="/auth", tags=["auth"])

# Simple in-memory brute-force protection (per IP, resets on restart)
# Production: use Redis-backed counter with TTL via Upstash
_login_attempts: dict[str, list[float]] = defaultdict(list)

# Per-endpoint limits (attempts per 60-second sliding window)
# login: 20/minute, register: 10/minute, refresh: 30/minute
_ENDPOINT_LIMITS: dict[str, int] = {
    "login": 20,
    "register": 10,
    "refresh": 30,
}
_WINDOW_SECONDS = 60  # 1-minute sliding window


def _check_rate_limit(ip: str, endpoint: str = "login") -> None:
    import time
    now = time.time()
    window_start = now - _WINDOW_SECONDS
    key = f"{ip}:{endpoint}"
    attempts = [t for t in _login_attempts[key] if t > window_start]
    _login_attempts[key] = attempts
    max_attempts = _ENDPOINT_LIMITS.get(endpoint, 20)
    if len(attempts) >= max_attempts:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Limit is {max_attempts}/minute.",
            headers={"Retry-After": "60"},
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
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)):
    # Same brute-force protection as /login — prevents enumeration and abuse
    _check_rate_limit(request.client.host if request.client else "unknown", "register")
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

    # Auto-create paper account for new user
    from app.models.account import Account
    paper_account = Account(
        user_id=user.id,
        broker="alpaca",
        mode="paper",
        label="Paper Account",
        extra_config={"equity": 100_000.0, "cash": 100_000.0},
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


@router.post("/demo", response_model=TokenResponse)
async def demo_login(db: AsyncSession = Depends(get_db)):
    """Issue a token for a shared demo user so the login-free public app is functional
    (every page/button needs a JWT). Gated by DEMO_MODE — disable for real multi-user.
    """
    if not settings.demo_mode:
        raise HTTPException(status_code=404, detail="Demo mode is disabled")
    import secrets as _secrets
    email = "demo@quantedge.app"
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


# ── Google OAuth ──────────────────────────────────────────────────────────────

@router.get("/google")
async def google_oauth_start():
    """Redirect to Google OAuth consent screen."""
    if not settings.google_client_id:
        raise HTTPException(status_code=503, detail="Google OAuth not configured")

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
    return RedirectResponse(url=url)


@router.get("/google/callback")
async def google_oauth_callback(
    code: str,
    db: AsyncSession = Depends(get_db),
):
    """Handle Google OAuth callback — create/login user and return tokens."""
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=503, detail="Google OAuth not configured")

    import httpx

    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    if token_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to exchange Google auth code")

    token_data = token_resp.json()

    # Get user info
    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
    if user_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to get Google user info")

    google_user = user_resp.json()
    email = google_user.get("email", "").lower()
    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email")

    # Find or create user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            # Random unusable password — Google users authenticate via OAuth only
            hashed_password=hash_password(str(uuid.uuid4())),
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    # Redirect to frontend with tokens in query params (SPA handles them)
    frontend_url = settings.cors_origins[0].strip()
    redirect_url = (
        f"{frontend_url}/auth/google/callback"
        f"?access_token={access_token}&refresh_token={refresh_token}"
    )
    return RedirectResponse(url=redirect_url)
