from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.utils.exceptions import UnauthorizedError
from app.utils.security import decode_token

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class _GuestUser:
    """Lightweight stand-in for a real User in demo/public access mode."""
    id: str = "00000000-0000-0000-0000-000000000001"
    email: str = "guest@quantedge.demo"
    is_active: bool = True
    is_superuser: bool = False
    hashed_password: str = ""

    # SQLAlchemy-like helpers expected by some endpoints
    def __getattr__(self, name: str) -> Any:
        return None


_GUEST: _GuestUser | None = None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | _GuestUser:
    # Demo mode: no token required — return a read-only guest user
    if not credentials and settings.demo_mode:
        global _GUEST
        if _GUEST is None:
            _GUEST = _GuestUser()
        return _GUEST  # type: ignore[return-value]

    if not credentials:
        raise UnauthorizedError()
    try:
        payload = decode_token(credentials.credentials)
        user_id: str = payload.get("sub")
        if not user_id or payload.get("type") != "access":
            raise UnauthorizedError()
    except JWTError:
        raise UnauthorizedError()

    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    user = result.scalar_one_or_none()
    if not user:
        raise UnauthorizedError()
    return user


async def get_current_active_superuser(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Superuser required")
    return current_user
