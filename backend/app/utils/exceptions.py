from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)


class NotFoundError(HTTPException):
    def __init__(
        self,
        detail: str = "Resource not found",
        signal_count: int | None = None,
        execution_time: float | None = None,
        pnl: float | None = None,
    ):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
        logger.info(
            "NotFoundError raised",
            extra={
                "detail": detail,
                "signal_count": signal_count,
                "execution_time": execution_time,
                "pnl": pnl,
            },
        )


class UnauthorizedError(HTTPException):
    def __init__(
        self,
        detail: str = "Not authenticated",
        signal_count: int | None = None,
        execution_time: float | None = None,
        pnl: float | None = None,
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )
        logger.info(
            "UnauthorizedError raised",
            extra={
                "detail": detail,
                "signal_count": signal_count,
                "execution_time": execution_time,
                "pnl": pnl,
            },
        )


class ForbiddenError(HTTPException):
    def __init__(
        self,
        detail: str = "Insufficient permissions",
        signal_count: int | None = None,
        execution_time: float | None = None,
        pnl: float | None = None,
    ):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
        logger.info(
            "ForbiddenError raised",
            extra={
                "detail": detail,
                "signal_count": signal_count,
                "execution_time": execution_time,
                "pnl": pnl,
            },
        )


class ConflictError(HTTPException):
    def __init__(
        self,
        detail: str = "Conflict",
        signal_count: int | None = None,
        execution_time: float | None = None,
        pnl: float | None = None,
    ):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)
        logger.info(
            "ConflictError raised",
            extra={
                "detail": detail,
                "signal_count": signal_count,
                "execution_time": execution_time,
                "pnl": pnl,
            },
        )


class BadRequestError(HTTPException):
    def __init__(
        self,
        detail: str = "Bad request",
        signal_count: int | None = None,
        execution_time: float | None = None,
        pnl: float | None = None,
    ):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
        logger.info(
            "BadRequestError raised",
            extra={
                "detail": detail,
                "signal_count": signal_count,
                "execution_time": execution_time,
                "pnl": pnl,
            },
        )


class BrokerError(HTTPException):
    def __init__(
        self,
        detail: str = "Broker API error",
        signal_count: int | None = None,
        execution_time: float | None = None,
        pnl: float | None = None,
    ):
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)
        logger.info(
            "BrokerError raised",
            extra={
                "detail": detail,
                "signal_count": signal_count,
                "execution_time": execution_time,
                "pnl": pnl,
            },
        )


class LiveTradingBlockedError(HTTPException):
    def __init__(
        self,
        signal_count: int | None = None,
        execution_time: float | None = None,
        pnl: float | None = None,
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is in paper mode. Switch to live mode to place real orders.",
        )
        logger.info(
            "LiveTradingBlockedError raised",
            extra={
                "detail": "Account is in paper mode. Switch to live mode to place real orders.",
                "signal_count": signal_count,
                "execution_time": execution_time,
                "pnl": pnl,
            },
        )