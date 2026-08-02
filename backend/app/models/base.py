from datetime import datetime, timezone
from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base  # noqa: F401 — re-export for all models


class TimestampMixin:
    """
    Mixin providing automatic ``created_at`` and ``updated_at`` timestamps.

    Handles edge cases:
    * ``None`` inputs – defaults to ``datetime.now(timezone.utc)``.
    * ``updated_at`` earlier than ``created_at`` – coerces to ``created_at``.
    * Empty collections are not relevant for timestamps but the mixin can be
      safely instantiated without positional arguments.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __init__(self, created_at: datetime | None = None, updated_at: datetime | None = None) -> None:
        """
        Initialise timestamp fields safely.

        Args:
            created_at: Optional creation timestamp. If ``None``, the current UTC
                time is used.
            updated_at: Optional update timestamp. If ``None``, the current UTC
                time is used. If provided and earlier than ``created_at``, it is
                adjusted to ``created_at`` to avoid off‑by‑one inconsistencies.
        """
        now = datetime.now(timezone.utc)

        # Guard against None inputs
        self.created_at = created_at if isinstance(created_at, datetime) else now
        self.updated_at = updated_at if isinstance(updated_at, datetime) else now

        # Off‑by‑one correction: ensure updated_at is not earlier than created_at
        if self.updated_at < self.created_at:
            self.updated_at = self.created_at

    @staticmethod
    def _ensure_datetime(value: datetime | None) -> datetime:
        """
        Convert ``None`` to the current UTC datetime.

        This helper isolates the logic for handling ``None`` inputs and can be
        reused by subclasses if they need to validate additional datetime fields.
        """
        return value if isinstance(value, datetime) else datetime.now(timezone.utc)