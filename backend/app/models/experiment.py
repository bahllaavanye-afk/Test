import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    val_accuracy: Mapped[float | None] = mapped_column(Numeric(6, 4))
    val_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    test_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    metrics_history: Mapped[list] = mapped_column(JSON, default=list)  # [{epoch, loss, acc}, ...]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
