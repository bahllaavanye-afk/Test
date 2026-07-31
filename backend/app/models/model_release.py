"""ModelRelease ORM — tracks every trained model artifact through its serving lifecycle."""
import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, Text, DateTime, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, validates
from app.database import Base
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

    @validates("model_name")
    def validate_model_name(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model_name must be a non-empty string.")
        if len(value) > 64:
            raise ValueError("model_name exceeds maximum length of 64 characters.")
        return value

    @validates("version")
    def validate_version(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("version must be a non-empty string.")
        if len(value) > 32:
            raise ValueError("version exceeds maximum length of 32 characters.")
        return value

    @validates("artifact_path")
    def validate_artifact_path(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("artifact_path must be a non-empty string.")
        if len(value) > 512:
            raise ValueError("artifact_path exceeds maximum length of 512 characters.")
        return value

    @validates("framework")
    def validate_framework(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("framework must be a non-empty string.")
        if len(value) > 32:
            raise ValueError("framework exceeds maximum length of 32 characters.")
        return value

    @validates("n_features", "seq_len")
    def validate_positive_int(self, key: str, value: int | None) -> int | None:
        if value is not None:
            if not isinstance(value, int):
                raise ValueError(f"{key} must be an integer if provided.")
            if value <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        return value

    @validates("traffic_pct")
    def validate_traffic_pct(self, key: str, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError("traffic_pct must be a numeric type.")
        if not (0.0 <= float(value) <= 100.0):
            raise ValueError("traffic_pct must be between 0 and 100 inclusive.")
        return float(value)

    @validates("created_by")
    def validate_created_by(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("created_by must be a non-empty string.")
        if len(value) > 128:
            raise ValueError("created_by exceeds maximum length of 128 characters.")
        return value