from fastapi import HTTPException, status


class NotFoundError(HTTPException):
    __slots__ = ()
    _default_instance = None

    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

    @classmethod
    def default(cls):
        """Return a cached instance for the default message."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance


class UnauthorizedError(HTTPException):
    __slots__ = ()
    _default_instance = None

    def __init__(self, detail: str = "Not authenticated"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

    @classmethod
    def default(cls):
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance


class ForbiddenError(HTTPException):
    __slots__ = ()
    _default_instance = None

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

    @classmethod
    def default(cls):
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance


class ConflictError(HTTPException):
    __slots__ = ()
    _default_instance = None

    def __init__(self, detail: str = "Conflict"):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)

    @classmethod
    def default(cls):
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance


class BadRequestError(HTTPException):
    __slots__ = ()
    _default_instance = None

    def __init__(self, detail: str = "Bad request"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    @classmethod
    def default(cls):
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance


class BrokerError(HTTPException):
    __slots__ = ()
    _default_instance = None

    def __init__(self, detail: str = "Broker API error"):
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)

    @classmethod
    def default(cls):
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance


class LiveTradingBlockedError(HTTPException):
    __slots__ = ()
    _default_instance = None

    def __init__(self):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is in paper mode. Switch to live mode to place real orders.",
        )

    @classmethod
    def default(cls):
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance


__all__ = [
    "NotFoundError",
    "UnauthorizedError",
    "ForbiddenError",
    "ConflictError",
    "BadRequestError",
    "BrokerError",
    "LiveTradingBlockedError",
]