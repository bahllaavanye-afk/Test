"""Task model for employee dispatch system."""
import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime, Float, String, Text
from sqlalchemy import Enum as SAEnum

try:
    from sqlalchemy.dialects.postgresql import JSONB as _JSONB
    _JsonType = _JSONB
except ImportError:  # pragma: no cover
    _JsonType = JSON  # type: ignore[assignment]

from app.database import Base


class TaskStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class TaskPriority(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String(50), nullable=False)
    assigned_to = Column(String(100), nullable=True)
    assigned_by = Column(String(100), nullable=True)
    status = Column(SAEnum(TaskStatus), default=TaskStatus.queued, nullable=False)
    priority = Column(SAEnum(TaskPriority), default=TaskPriority.medium, nullable=False)
    params = Column(_JsonType, default=dict)
    result = Column(_JsonType, nullable=True)
    error_message = Column(Text, nullable=True)
    progress_pct = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    due_at = Column(DateTime(timezone=True), nullable=True)
