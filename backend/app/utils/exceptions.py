"""Utility exception classes for the FastAPI backend.

This module defines a collection of custom HTTPException subclasses used
throughout the backend to provide clear, typed error responses. Each class
encapsulates a specific HTTP status code and a default error message, but
the message can be overridden when raising the exception.

The exceptions are lightweight wrappers around ``fastapi.HTTPException`` and
do not add any additional behavior beyond documentation and type hints.
"""

from fastapi import HTTPException, status


class NotFoundError(HTTPException):
    """Exception raised when a requested resource cannot be found (HTTP 404).

    Parameters
    ----------
    detail: str, optional
        Human‑readable description of the error. Defaults to ``"Resource not found"``.
    """

    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class UnauthorizedError(HTTPException):
    """Exception raised when authentication is required but missing or invalid (HTTP 401).

    Parameters
    ----------
    detail: str, optional
        Human‑readable description of the error. Defaults to ``"Not authenticated"``.
    """

    def __init__(self, detail: str = "Not authenticated") -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenError(HTTPException):
    """Exception raised when the authenticated user lacks sufficient permissions (HTTP 403).

    Parameters
    ----------
    detail: str, optional
        Human‑readable description of the error. Defaults to ``"Insufficient permissions"``.
    """

    def __init__(self, detail: str = "Insufficient permissions") -> None:
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class ConflictError(HTTPException):
    """Exception raised when a request conflicts with the current state of the resource (HTTP 409).

    Parameters
    ----------
    detail: str, optional
        Human‑readable description of the error. Defaults to ``"Conflict"``.
    """

    def __init__(self, detail: str = "Conflict") -> None:
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)


class BadRequestError(HTTPException):
    """Exception raised for malformed client requests (HTTP 400).

    Parameters
    ----------
    detail: str, optional
        Human‑readable description of the error. Defaults to ``"Bad request"``.
    """

    def __init__(self, detail: str = "Bad request") -> None:
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class BrokerError(HTTPException):
    """Exception raised when an upstream broker API returns an error (HTTP 502).

    Parameters
    ----------
    detail: str, optional
        Human‑readable description of the error. Defaults to ``"Broker API error"``.
    """

    def __init__(self, detail: str = "Broker API error") -> None:
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)


class LiveTradingBlockedError(HTTPException):
    """Exception raised when attempting to place real orders while the account is in paper mode (HTTP 403)."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is in paper mode. Switch to live mode to place real orders.",
        )