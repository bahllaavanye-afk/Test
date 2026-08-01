import uuid
from datetime import datetime
from typing import List, Dict, Optional, Union

from sqlalchemy import String, Numeric, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Experiment(Base):
    """
    SQLAlchemy model representing a machine‑learning experiment.

    Attributes
    ----------
    id : Mapped[str]
        Primary key, generated as a UUID string.
    name : Mapped[str]
        Human‑readable unique name for the experiment.
    config : Mapped[Dict]
        JSON‑serialisable configuration dictionary used to launch the experiment.
    status : Mapped[str]
        Current lifecycle status (``queued``, ``running``, ``done`` or ``failed``).
    val_accuracy : Mapped[Optional[float]]
        Validation accuracy recorded after training; ``None`` if not yet available.
    val_sharpe : Mapped[Optional[float]]
        Validation Sharpe ratio; ``None`` if not yet computed.
    test_sharpe : Mapped[Optional[float]]
        Sharpe ratio on the test set; ``None`` if not yet computed.
    artifact_path : Mapped[Optional[str]]
        Filesystem path to the stored artefacts (model weights, logs, etc.).
    started_at : Mapped[Optional[datetime]]
        Timestamp when the experiment started execution.
    completed_at : Mapped[Optional[datetime]]
        Timestamp when the experiment finished (successfully or with failure).
    error_message : Mapped[Optional[str]]
        Error details if the experiment failed; otherwise ``None``.
    metrics_history : Mapped[List[Dict]]
        Chronological list of metric dictionaries (e.g., ``{'epoch': 1, 'loss': 0.5, 'acc': 0.8}``).
    created_at : Mapped[datetime]
        Record creation timestamp.
    """

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    config: Mapped[Dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="queued"
    )  # queued|running|done|failed
    val_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    val_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    test_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    metrics_history: Mapped[List[Dict]] = mapped_column(
        JSON, default=list
    )  # [{epoch, loss, acc}, ...]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )