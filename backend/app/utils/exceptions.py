from fastapi import HTTPException, status
from typing import Any


def _sanitize_detail(detail: Any, default: str) -> str:
    """
    Ensure that the detail provided to an HTTPException is a non‑empty string.
    Handles None, empty strings, and empty collections by falling back to a default message.
    """
    if detail is None:
        return default
    # Convert non‑string details to string, but treat empty collections specially
    if isinstance(detail, (list, dict, set, tuple)):
        return default if len(detail) == 0 else str(detail)
    # Strip whitespace for strings; treat empty after stripping as missing
    if isinstance(detail, str):
        stripped = detail.strip()
        return default if not stripped else stripped
    # Fallback for any other type
    return str(detail) if str(detail) else default


class NotFoundError(HTTPException):
    def __init__(self, detail: Any = None):
        sanitized_detail = _sanitize_detail(detail, "Resource not found")
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=sanitized_detail)


class UnauthorizedError(HTTPException):
    def __init__(self, detail: Any = None):
        sanitized_detail = _sanitize_detail(detail, "Not authenticated")
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=sanitized_detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenError(HTTPException):
    def __init__(self, detail: Any = None):
        sanitized_detail = _sanitize_detail(detail, "Insufficient permissions")
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=sanitized_detail)


class ConflictError(HTTPException):
    def __init__(self, detail: Any = None):
        sanitized_detail = _sanitize_detail(detail, "Conflict")
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=sanitized_detail)


class BadRequestError(HTTPException):
    def __init__(self, detail: Any = None):
        sanitized_detail = _sanitize_detail(detail, "Bad request")
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitized_detail)


class BrokerError(HTTPException):
    def __init__(self, detail: Any = None):
        sanitized_detail = _sanitize_detail(detail, "Broker API error")
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=sanitized_detail)


class LiveTradingBlockedError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is in paper mode. Switch to live mode to place real orders.",
        )