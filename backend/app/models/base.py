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


# ==============================
# Unit tests for TimestampMixin
# ==============================
def _get_column(model, attr_name):
    """Helper to retrieve the underlying Column object for a mapped attribute."""
    return getattr(model, attr_name).property.columns[0]


def test_timestamp_mixin_created_at_properties():
    """
    Verify that the `created_at` column is timezone aware,
    non‑nullable, and has a server default.
    """
    col = _get_column(TimestampMixin, "created_at")
    # Ensure the column type includes timezone information
    assert isinstance(col.type, DateTime)
    assert col.type.timezone is True, "created_at should be timezone aware"
    # Column must be defined as NOT NULL
    assert col.nullable is False, "created_at should be non‑nullable"
    # Server default must be present (func.now())
    assert col.server_default is not None, "created_at should have a server default"


def test_timestamp_mixin_updated_at_properties():
    """
    Verify that the `updated_at` column mirrors `created_at` in timezone awareness
    and nullability, and also defines an on‑update clause.
    """
    col = _get_column(TimestampMixin, "updated_at")
    # Ensure the column type includes timezone information
    assert isinstance(col.type, DateTime)
    assert col.type.timezone is True, "updated_at should be timezone aware"
    # Column must be defined as NOT NULL
    assert col.nullable is False, "updated_at should be non‑nullable"
    # Server default must be present (func.now())
    assert col.server_default is not None, "updated_at should have a server default"
    # onupdate clause should be defined (func.now())
    assert col.onupdate is not None, "updated_at should have an onupdate clause"


def test_timestamp_mixin_column_consistency():
    """
    Edge‑case test: ensure both timestamp columns use the same underlying SQL type
    and share the same timezone setting.
    """
    created_col = _get_column(TimestampMixin, "created_at")
    updated_col = _get_column(TimestampMixin, "updated_at")
    assert type(created_col.type) is type(updated_col.type), "Both columns should share the same type"
    assert created_col.type.timezone == updated_col.type.timezone, "Timezone setting should be identical for both columns"