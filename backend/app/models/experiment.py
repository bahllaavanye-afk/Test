import uuid
from datetime import datetime
from typing import List, Dict, Optional, Literal

from sqlalchemy import String, Numeric, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column
from pydantic import BaseModel, Field, validator, root_validator

from app.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="queued"
    )  # queued|running|done|failed
    val_accuracy: Mapped[float | None] = mapped_column(Numeric(6, 4))
    val_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    test_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    metrics_history: Mapped[list] = mapped_column(JSON, default=list)  # [{epoch, loss, acc}, ...]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


# ----------------------------------------------------------------------
# Pydantic schemas for API interaction
# ----------------------------------------------------------------------


class ExperimentBase(BaseModel):
    name: str = Field(
        ...,
        description="Human‑readable identifier for the experiment; must be unique.",
        example="mean_rev_20_2_run_01",
        max_length=128,
    )
    config: Dict = Field(
        ...,
        description="Configuration dictionary describing hyper‑parameters, data sources, and model settings.",
        example={"window": 20, "threshold": 2.0},
    )
    status: Literal["queued", "running", "done", "failed"] = Field(
        "queued",
        description="Current lifecycle state of the experiment.",
        example="queued",
    )
    val_accuracy: Optional[float] = Field(
        None,
        description="Validation accuracy metric (0‑1 range).",
        example=0.8425,
        ge=0.0,
        le=1.0,
    )
    val_sharpe: Optional[float] = Field(
        None,
        description="Validation Sharpe ratio.",
        example=1.34,
    )
    test_sharpe: Optional[float] = Field(
        None,
        description="Test set Sharpe ratio.",
        example=1.21,
    )
    artifact_path: Optional[str] = Field(
        None,
        description="Filesystem path or object store URI where experiment artifacts are stored.",
        example="/mnt/artifacts/exp_12345/",
        max_length=512,
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the experiment started execution.",
        example="2024-09-01T12:34:56Z",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the experiment finished execution.",
        example="2024-09-01T14:20:10Z",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message captured if the experiment failed.",
        example="Division by zero in metric calculation.",
    )
    metrics_history: List[Dict] = Field(
        default_factory=list,
        description="Chronological list of metric snapshots per training epoch.",
        example=[{"epoch": 1, "loss": 0.45, "acc": 0.78}],
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of experiment record creation.",
        example="2024-09-01T12:00:00Z",
    )

    @validator("name")
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Experiment name must not be empty or whitespace.")
        return v

    @validator("status")
    def status_must_be_valid(cls, v: str) -> str:
        allowed = {"queued", "running", "done", "failed"}
        if v not in allowed:
            raise ValueError(f"Status '{v}' is not one of {allowed}.")
        return v

    @validator("metrics_history", each_item=True)
    def metric_entry_structure(cls, v: dict) -> dict:
        required_keys = {"epoch", "loss"}
        if not required_keys.issubset(v.keys()):
            missing = required_keys - v.keys()
            raise ValueError(f"Metrics entry missing required keys: {missing}")
        return v

    @root_validator
    def artifact_path_required_when_done(cls, values):
        status = values.get("status")
        artifact_path = values.get("artifact_path")
        if status == "done" and not artifact_path:
            raise ValueError("artifact_path must be set when status is 'done'.")
        return values

    class Config:
        orm_mode = True
        anystr_strip_whitespace = True


class ExperimentCreate(ExperimentBase):
    """Schema for creating a new experiment. All fields except optional ones are required."""

    pass


class ExperimentRead(ExperimentBase):
    """Schema for reading experiment details, includes the generated identifier."""

    id: str = Field(
        ...,
        description="Unique identifier of the experiment (UUID).",
        example="550e8400-e29b-41d4-a716-446655440000",
    )


class ExperimentUpdate(BaseModel):
    """Schema for partial updates to an existing experiment."""

    name: Optional[str] = Field(
        None,
        description="Optional new name for the experiment; must be unique if provided.",
        max_length=128,
    )
    config: Optional[Dict] = Field(
        None,
        description="Optional new configuration dictionary.",
    )
    status: Optional[Literal["queued", "running", "done", "failed"]] = Field(
        None,
        description="Optional new status value.",
    )
    val_accuracy: Optional[float] = Field(
        None,
        description="Optional updated validation accuracy (0‑1).",
        ge=0.0,
        le=1.0,
    )
    val_sharpe: Optional[float] = Field(
        None,
        description="Optional updated validation Sharpe ratio.",
    )
    test_sharpe: Optional[float] = Field(
        None,
        description="Optional updated test Sharpe ratio.",
    )
    artifact_path: Optional[str] = Field(
        None,
        description="Optional updated artifact storage path.",
        max_length=512,
    )
    error_message: Optional[str] = Field(
        None,
        description="Optional error message if the experiment encountered a failure.",
    )
    metrics_history: Optional[List[Dict]] = Field(
        None,
        description="Optional replacement of the metrics history list.",
    )

    @validator("name")
    def name_not_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Experiment name must not be empty or whitespace.")
        return v

    @validator("status")
    def status_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            allowed = {"queued", "running", "done", "failed"}
            if v not in allowed:
                raise ValueError(f"Status '{v}' is not one of {allowed}.")
        return v

    @validator("metrics_history", each_item=True)
    def metric_entry_structure(cls, v: dict) -> dict:
        required_keys = {"epoch", "loss"}
        if not required_keys.issubset(v.keys()):
            missing = required_keys - v.keys()
            raise ValueError(f"Metrics entry missing required keys: {missing}")
        return v

    class Config:
        anystr_strip_whitespace = True
        orm_mode = True