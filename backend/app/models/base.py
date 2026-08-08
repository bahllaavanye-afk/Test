from datetime import datetime, timezone
from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base  # noqa: F401 — re-export for all models


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    @staticmethod
    def validate_timestamp(value: datetime, field_name: str = "timestamp") -> None:
        """
        Validate that a timestamp is a timezone-aware datetime instance.

        Parameters
        ----------
        value : datetime
            The timestamp value to validate.
        field_name : str, optional
            Name of the field being validated (used in error messages).

        Raises
        ------
        ValueError
            If `value` is not a datetime instance or is not timezone-aware.
        """
        if not isinstance(value, datetime):
            raise ValueError(f"{field_name} must be a datetime instance, got {type(value).__name__}")
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(f"{field_name} must be timezone-aware")