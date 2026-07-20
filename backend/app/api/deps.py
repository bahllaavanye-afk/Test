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