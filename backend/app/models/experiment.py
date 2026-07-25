import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import String, Numeric, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from pydantic import BaseModel, Field, validator

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


class ExperimentSchema(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier for the experiment.",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    name: str = Field(
        ...,
        max_length=128,
        description="Human‑readable unique name of the experiment.",
        example="mean_rev_20_2_run_01",
    )
    config: Dict[str, Any] = Field(
        ...,
        description="Configuration parameters for the experiment.",
        example={"window": 20, "threshold": 2},
    )
    status: str = Field(
        default="queued",
        description="Current status of the experiment. Allowed values: queued, running, done, failed.",
        example="running",
    )
    val_accuracy: Optional[float] = Field(
        None,
        description="Validation accuracy metric (0 – 1).",
        example=0.8423,
    )
    val_sharpe: Optional[float] = Field(
        None,
        description="Validation Sharpe ratio.",
        example=1.23,
    )
    test_sharpe: Optional[float] = Field(
        None,
        description="Test Sharpe ratio.",
        example=1.15,
    )
    artifact_path: Optional[str] = Field(
        None,
        max_length=512,
        description="Filesystem path to stored artifacts.",
        example="/mnt/artifacts/exp1",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the experiment started.",
        example="2023-01-01T12:00:00Z",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the experiment completed.",
        example="2023-01-01T14:00:00Z",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error details if the experiment failed.",
    )
    metrics_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical metrics per epoch.",
        example=[{"epoch": 1, "loss": 0.45, "acc": 0.78}],
    )
    created_at: datetime = Field(
        ...,
        description="Record creation timestamp.",
        example="2023-01-01T11:55:00Z",
    )

    @validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"queued", "running", "done", "failed"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v

    @validator("val_accuracy")
    def validate_val_accuracy(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("val_accuracy must be between 0 and 1")
        return v

    @validator("metrics_history", each_item=True)
    def validate_metrics_item(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = {"epoch"}
        if not isinstance(v, dict):
            raise TypeError("Each item in metrics_history must be a dict")
        missing = required_keys - v.keys()
        if missing:
            raise ValueError(f"Metrics entry missing required keys: {missing}")
        return v

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "mean_rev_20_2_run_01",
                "config": {"window": 20, "threshold": 2},
                "status": "queued",
                "val_accuracy": None,
                "val_sharpe": None,
                "test_sharpe": None,
                "artifact_path": None,
                "started_at": None,
                "completed_at": None,
                "error_message": None,
                "metrics_history": [],
                "created_at": "2023-01-01T11:55:00Z",
            }
        }