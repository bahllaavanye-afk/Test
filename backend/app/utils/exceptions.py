from fastapi import HTTPException, status
from typing import Optional


def _resolve_detail(provided_detail: Optional[str], default_detail: str) -> str:
    """Return a valid detail string, falling back to a default if the input is None or empty."""
    if isinstance(provided_detail, str) and provided_detail.strip():
        return provided_detail
    return default_detail


class NotFoundError(HTTPException):
    def __init__(self, detail: Optional[str] = None):
        resolved_detail = _resolve_detail(detail, "Resource not found")
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=resolved_detail)


class UnauthorizedError(HTTPException):
    def __init__(self, detail: Optional[str] = None):
        resolved_detail = _resolve_detail(detail, "Not authenticated")
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=resolved_detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenError(HTTPException):
    def __init__(self, detail: Optional[str] = None):
        resolved_detail = _resolve_detail(detail, "Insufficient permissions")
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=resolved_detail)


class ConflictError(HTTPException):
    def __init__(self, detail: Optional[str] = None):
        resolved_detail = _resolve_detail(detail, "Conflict")
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=resolved_detail)


class BadRequestError(HTTPException):
    def __init__(self, detail: Optional[str] = None):
        resolved_detail = _resolve_detail(detail, "Bad request")
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=resolved_detail)


class BrokerError(HTTPException):
    def __init__(self, detail: Optional[str] = None):
        resolved_detail = _resolve_detail(detail, "Broker API error")
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=resolved_detail)


class LiveTradingBlockedError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is in paper mode. Switch to live mode to place real orders.",
        )