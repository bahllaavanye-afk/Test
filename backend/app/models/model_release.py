"""ModelRelease ORM — tracks every trained model artifact through its serving lifecycle."""
import uuid
import functools
from datetime import datetime
from sqlalchemy import String, Float, Integer, Text, DateTime, JSON, Index, event
from sqlalchemy.orm import Mapped, mapped_column, Session
from app.database import Base, SessionLocal
from app.models.base import TimestampMixin


class ModelRelease(Base, TimestampMixin):
    """
    One row per trained model artifact registered for serving.

    Lifecycle:
        registered → shadow → challenger → champion → archived

    Only one champion and one challenger per (model_name) are allowed at a time.
    Promoting a challenger to champion atomically archives the old champion.
    """
    __tablename__ = "model_releases"
    __table_args__ = (
        Index("ix_mr_model_status", "model_name", "status"),
    )

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Logical model name shared across versions, e.g. "lstm_momentum"
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Artifact version, e.g. "v1.0.0" or "20240115_001"
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    # Filesystem path to the serialized model weights / pickle
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    # Serialization framework
    framework: Mapped[str] = mapped_column(String(32), nullable=False, default="pytorch")
    # Model architecture size params
    n_features: Mapped[int | None] = mapped_column(Integer)
    seq_len: Mapped[int | None] = mapped_column(Integer)
    # JSON blobs
    model_params: Mapped[dict] = mapped_column(JSON, default=dict)
    training_config: Mapped[dict] = mapped_column(JSON, default=dict)
    train_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    live_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    # Serving state
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="registered", index=True
    )
    # % of inference traffic routed to this release when it's a challenger (0–100)
    traffic_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Free-text notes from the creator
    notes: Mapped[str | None] = mapped_column(Text)
    # Timestamps for key lifecycle events
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Who registered this release (email or "system")
    created_by: Mapped[str] = mapped_column(String(128), nullable=False, default="system")

    @classmethod
    @functools.lru_cache(maxsize=256)
    def get_champion_id(cls, model_name: str) -> str | None:
        """
        Retrieve the primary key of the current champion for a given model name.

        This method caches the result, avoiding repeated expensive database scans.
        Cache is cleared automatically on inserts/updates to ModelRelease.
        """
        with SessionLocal() as session:  # type: Session
            result = (
                session.query(cls.id)
                .filter(cls.model_name == model_name, cls.status == "champion")
                .order_by(cls.promoted_at.desc())
                .first()
            )
            return result[0] if result else None

    @classmethod
    def invalidate_champion_cache(cls) -> None:
        """Explicitly clear the champion‑id cache."""
        cls.get_champion_id.cache_clear()


# Invalidate the champion cache whenever a ModelRelease row is inserted or updated.
@event.listens_for(ModelRelease, "after_insert")
@event.listens_for(ModelRelease, "after_update")
def _clear_champion_cache(mapper, connection, target):
    ModelRelease.invalidate_champion_cache()