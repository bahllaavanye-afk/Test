"""Dependency utilities for FastAPI authentication.

This module provides dependency functions used across the API to retrieve the
currently authenticated user and to enforce super‑user privileges. The
functions rely on JWT decoding, SQLAlchemy async sessions, and custom exception
handling defined elsewhere in the codebase.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import MultipleResultsFound
from app.database import get_db
from app.models.user import User
from app.utils.security import decode_token
from app.utils.exceptions import UnauthorizedError

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession | None = Depends(get_db),
) -> User:
    """Retrieve the authenticated user from a JWT token.

    Args:
        credentials: HTTP bearer credentials extracted from the request header.
            If ``None`` or missing a token, an ``UnauthorizedError`` is raised.
        db: Async SQLAlchemy session used to query the ``User`` table. Must be
            provided; otherwise an ``UnauthorizedError`` is raised.

    Returns:
        The ``User`` instance corresponding to the token's ``sub`` claim.

    Raises:
        UnauthorizedError: If credentials are missing/invalid, the token cannot
            be decoded, the user is not found, or the database query returns an
            unexpected result.
    """
    # Handle missing credentials
    if not credentials:
        raise UnauthorizedError()
    # Handle empty token string
    token = credentials.credentials
    if not token:
        raise UnauthorizedError()
    # Decode token and validate payload
    try:
        payload = decode_token(token)
        user_id: str | None = payload.get("sub")
        if not user_id or payload.get("type") != "access":
            raise UnauthorizedError()
    except JWTError:
        raise UnauthorizedError()
    # Validate DB session
    if db is None:
        raise UnauthorizedError()
    # Query user, handling empty or unexpected result sets
    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    try:
        user = result.scalar_one_or_none()
    except MultipleResultsFound:
        raise UnauthorizedError()
    if not user:
        raise UnauthorizedError()
    return user


async def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure the current user has super‑user privileges.

    Args:
        current_user: The authenticated ``User`` instance obtained via
            ``get_current_user`` dependency.

    Returns:
        The same ``User`` instance if it is a superuser.

    Raises:
        HTTPException: With status code 403 if the user is not a superuser.
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Superuser required")
    return current_user