import logging
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

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession | None = Depends(get_db),
) -> User:
    # Handle missing credentials
    if not credentials:
        logger.error("Missing authentication credentials")
        raise UnauthorizedError()
    # Handle empty token string
    token = credentials.credentials
    if not token:
        logger.error("Empty authentication token provided")
        raise UnauthorizedError()
    # Decode token and validate payload
    try:
        payload = decode_token(token)
        user_id: str | None = payload.get("sub")
        if not user_id or payload.get("type") != "access":
            logger.error(
                "Invalid token payload: missing user_id or incorrect token type",
                extra={"user_id": user_id, "token_type": payload.get("type")},
            )
            raise UnauthorizedError()
    except JWTError as e:
        logger.error("JWT decoding failed", exc_info=e)
        raise UnauthorizedError()
    # Validate DB session
    if db is None:
        logger.error("Database session is None")
        raise UnauthorizedError()
    # Query user, handling empty or unexpected result sets
    try:
        result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
        user = result.scalar_one_or_none()
    except MultipleResultsFound as e:
        logger.error(
            "Multiple users found for user_id",
            extra={"user_id": user_id},
            exc_info=e,
        )
        raise UnauthorizedError()
    except Exception as e:
        logger.error(
            "Database query failed while fetching user",
            extra={"user_id": user_id},
            exc_info=e,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    if not user:
        logger.error("Authenticated user not found or inactive", extra={"user_id": user_id})
        raise UnauthorizedError()
    return user


async def get_current_active_superuser(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_superuser:
        logger.error(
            "Superuser permission required",
            extra={"user_id": getattr(current_user, "id", None)},
        )
        raise HTTPException(status_code=403, detail="Superuser required")
    return current_user