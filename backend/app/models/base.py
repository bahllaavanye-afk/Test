from datetime import datetime, timezone
from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base  # noqa: F401 — re-export for all models


class TimestampMixin:
    """
    Mixin providing ``created_at`` and ``updated_at`` columns with sensible defaults.
    Includes helper methods that safely handle ``None`` inputs and guard against
    off‑by‑one timestamp anomalies (e.g., ``updated_at`` earlier than ``created_at``).
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def set_timestamps(
        self,
        created: datetime | None = None,
        updated: datetime | None = None,
    ) -> None:
        """
        Safely assign timestamps.

        * If *created* is ``None``, the current UTC time is used.
        * If *updated* is ``None``, it defaults to *created* (or current UTC time if *created* is also ``None``).
        * Guarantees that ``updated_at`` is not earlier than ``created_at``; if it is,
          ``updated_at`` is set to ``created_at`` to avoid off‑by‑one inconsistencies.
        """
        now = datetime.now(timezone.utc)

        # Resolve created timestamp
        if created is None:
            created = now
        elif created.tzinfo is None:
            # Assume naive datetimes are UTC
            created = created.replace(tzinfo=timezone.utc)

        # Resolve updated timestamp
        if updated is None:
            updated = created
        elif updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)

        # Guard against updated earlier than created
        if updated < created:
            updated = created

        self.created_at = created
        self.updated_at = updated

    @classmethod
    def from_dict(cls, data: dict | None = None, **overrides):
        """
        Create an instance of the model (or subclass) from a dictionary,
        handling ``None`` or empty inputs gracefully.

        Parameters
        ----------
        data: dict | None
            Source mapping with field values. ``None`` or an empty dict results
            in an instance with default timestamps only.
        **overrides:
            Additional keyword arguments that take precedence over ``data``.

        Returns
        -------
        cls
            An instantiated object with timestamps set appropriately.
        """
        if not data:
            data = {}

        # Merge overrides, giving them priority
        init_kwargs = {**data, **overrides}

        # Extract timestamp fields if present; otherwise let ``set_timestamps`` handle defaults
        created = init_kwargs.pop("created_at", None)
        updated = init_kwargs.pop("updated_at", None)

        instance = cls(**init_kwargs)  # type: ignore[arg-type]  # Subclass may accept arbitrary kwargs
        instance.set_timestamps(created=created, updated=updated)
        return instance

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"created_at={self.created_at.isoformat()}, "
            f"updated_at={self.updated_at.isoformat()}>"
        )