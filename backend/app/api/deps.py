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


async def get_current_active_superuser(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Superuser required")
    return current_user


# ===========================
# Unit tests for edge cases
# ===========================
import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.exc import MultipleResultsFound as SA_MultipleResultsFound


@pytest.mark.asyncio
async def test_missing_credentials_raises_unauthorized():
    """When credentials are None, UnauthorizedError should be raised."""
    mock_db = AsyncMock()
    with pytest.raises(UnauthorizedError):
        await get_current_user(credentials=None, db=mock_db)


@pytest.mark.asyncio
async def test_empty_token_string_raises_unauthorized():
    """When token string is empty, UnauthorizedError should be raised."""
    mock_db = AsyncMock()
    empty_credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
    with pytest.raises(UnauthorizedError):
        await get_current_user(credentials=empty_credentials, db=mock_db)


@pytest.mark.asyncio
async def test_invalid_token_type_raises_unauthorized(monkeypatch):
    """Token payload with a non‑access type should raise UnauthorizedError."""
    mock_db = AsyncMock()
    valid_credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="dummy-token")

    def fake_decode_token(token: str):
        return {"sub": "user-123", "type": "refresh"}  # wrong type

    monkeypatch.setattr("app.utils.security.decode_token", fake_decode_token)

    with pytest.raises(UnauthorizedError):
        await get_current_user(credentials=valid_credentials, db=mock_db)


@pytest.mark.asyncio
async def test_multiple_results_found_raises_unauthorized(monkeypatch):
    """If the query returns multiple users, UnauthorizedError should be raised."""
    # Mock DB execute to raise MultipleResultsFound
    mock_db = AsyncMock()
    mock_db.execute.side_effect = SA_MultipleResultsFound

    valid_credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")

    def fake_decode_token(token: str):
        return {"sub": "user-duplicate", "type": "access"}

    monkeypatch.setattr("app.utils.security.decode_token", fake_decode_token)

    with pytest.raises(UnauthorizedError):
        await get_current_user(credentials=valid_credentials, db=mock_db)