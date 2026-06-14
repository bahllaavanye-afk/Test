"""Task model for employee dispatch system."""
import uuid
from datetime import UTC, datetime
try:
    from sqlalchemy.dialects.postgresql import JSONB
except ImportError:
    from sqlalchemy import JSON as JSONB  # type: ignore[no-redef]
from sqlalchemy import Column, String, Text, DateTime, Enum as SAEnum, Float
from app.database import Base
import enum


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
    params = Column(JSONB, default=dict)
    result = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    progress_pct = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    due_at = Column(DateTime(timezone=True), nullable=True)
