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
    """
    Retrieve the currently authenticated user from the JWT token.

    Parameters
    ----------
    credentials: HTTPAuthorizationCredentials | None
        The bearer token extracted from the request header. If ``None`` the request
        is considered unauthenticated.
    db: AsyncSession | None
        An asynchronous SQLAlchemy session provided by the ``get_db`` dependency.
        If ``None`` the function cannot query the database.

    Returns
    -------
    User
        The active ``User`` instance corresponding to the token's ``sub`` claim.

    Raises
    ------
    UnauthorizedError
        If the token is missing, malformed, does not represent an access token,
        cannot be decoded, the user does not exist, or any database inconsistency
        occurs.
    """
    # Handle missing credentials
    if not credentials:
        raise UnauthorizedError()
    # Handle empty token string
    token: str = credentials.credentials
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
        user: User | None = result.scalar_one_or_none()
    except MultipleResultsFound:
        raise UnauthorizedError()
    if not user:
        raise UnauthorizedError()
    return user


async def get_current_active_superuser(current_user: User = Depends(get_current_user)) -> User:
    """
    Ensure that the current user is an active superuser.

    Parameters
    ----------
    current_user: User
        The user object obtained from ``get_current_user`` dependency.

    Returns
    -------
    User
        The same ``User`` instance if it has superuser privileges.

    Raises
    ------
    HTTPException
        With status code 403 if the user is not a superuser.
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Superuser required")
    return current_user