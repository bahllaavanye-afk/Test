"""Custom HTTP exception classes for the QuantEdge API."""

from fastapi import HTTPException, status

__all__ = [
    "NotFoundError",
    "UnauthorizedError",
    "ForbiddenError",
    "ConflictError",
    "BadRequestError",
    "BrokerError",
    "LiveTradingBlockedError",
]


class NotFoundError(HTTPException):
    """Exception for resources that cannot be found (HTTP 404)."""

    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class UnauthorizedError(HTTPException):
    """Exception for unauthenticated access attempts (HTTP 401)."""

    def __init__(self, detail: str = "Not authenticated"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenError(HTTPException):
    """Exception for insufficient permissions (HTTP 403)."""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class ConflictError(HTTPException):
    """Exception for request conflicts (HTTP 409)."""

    def __init__(self, detail: str = "Conflict"):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)


class BadRequestError(HTTPException):
    """Exception for malformed requests (HTTP 400)."""

    def __init__(self, detail: str = "Bad request"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class BrokerError(HTTPException):
    """Exception for errors returned by broker APIs (HTTP 502)."""

    def __init__(self, detail: str = "Broker API error"):
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)


class LiveTradingBlockedError(HTTPException):
    """Exception raised when attempting live trades while in paper mode."""

    def __init__(self):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is in paper mode. Switch to live mode to place real orders.",
        )