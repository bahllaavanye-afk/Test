"""ModelRelease ORM — tracks every trained model artifact through its serving lifecycle."""
import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, Text, DateTime, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base
from app.models.base import TimestampMixin

# Constants
TABLE_NAME = "model_releases"
INDEX_NAME = "ix_mr_model_status"

MODEL_NAME_MAX_LENGTH = 64
VERSION_MAX_LENGTH = 32
ARTIFACT_PATH_MAX_LENGTH = 512
FRAMEWORK_MAX_LENGTH = 32
STATUS_MAX_LENGTH = 16
CREATED_BY_MAX_LENGTH = 128

DEFAULT_FRAMEWORK = "pytorch"
DEFAULT_STATUS = "registered"
DEFAULT_TRAFFIC_PCT = 0.0
DEFAULT_CREATED_BY = "system"


class ModelRelease(Base, TimestampMixin):
    """
    One row per trained model artifact registered for serving.

    Lifecycle:
        registered → shadow → challenger → champion → archived

    Only one champion and one challenger per (model_name) are allowed at a time.
    Promoting a challenger to champion atomically archives the old champion.
    """
    __tablename__ = TABLE_NAME
    __table_args__ = (
        Index(INDEX_NAME, "model_name", "status"),
    )

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Logical model name shared across versions, e.g. "lstm_momentum"
    model_name: Mapped[str] = mapped_column(String(MODEL_NAME_MAX_LENGTH), nullable=False, index=True)
    # Artifact version, e.g. "v1.0.0" or "20240115_001"
    version: Mapped[str] = mapped_column(String(VERSION_MAX_LENGTH), nullable=False)
    # Filesystem path to the serialized model weights / pickle
    artifact_path: Mapped[str] = mapped_column(String(ARTIFACT_PATH_MAX_LENGTH), nullable=False)
    # Serialization framework
    framework: Mapped[str] = mapped_column(String(FRAMEWORK_MAX_LENGTH), nullable=False, default=DEFAULT_FRAMEWORK)
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
        String(STATUS_MAX_LENGTH), nullable=False, default=DEFAULT_STATUS, index=True
    )
    # % of inference traffic routed to this release when it's a challenger (0–100)
    traffic_pct: Mapped[float] = mapped_column(Float, nullable=False, default=DEFAULT_TRAFFIC_PCT)
    # Free-text notes from the creator
    notes: Mapped[str | None] = mapped_column(Text)
    # Timestamps for key lifecycle events
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Who registered this release (email or "system")
    created_by: Mapped[str] = mapped_column(String(CREATED_BY_MAX_LENGTH), nullable=False, default=DEFAULT_CREATED_BY)