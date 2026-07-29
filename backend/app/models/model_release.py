"""ModelRelease ORM — tracks every trained model artifact through its serving lifecycle."""
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Float, Index, Integer, JSON, String, Text, DateTime
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
    n_features: Mapped[Optional[int]] = mapped_column(Integer)
    seq_len: Mapped[Optional[int]] = mapped_column(Integer)
    # JSON blobs
    model_params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    training_config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    train_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    live_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    # Serving state
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="registered", index=True
    )
    # % of inference traffic routed to this release when it's a challenger (0–100)
    traffic_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Free-text notes from the creator
    notes: Mapped[Optional[str]] = mapped_column(Text)
    # Timestamps for key lifecycle events
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    # Who registered this release (email or "system")
    created_by: Mapped[str] = mapped_column(String(128), nullable=False, default="system")

    def __init__(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        framework: str = "pytorch",
        n_features: Optional[int] = None,
        seq_len: Optional[int] = None,
        model_params: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        train_metrics: Optional[Dict[str, Any]] = None,
        live_metrics: Optional[Dict[str, Any]] = None,
        status: str = "registered",
        traffic_pct: float = 0.0,
        notes: Optional[str] = None,
        created_by: str = "system",
        **kwargs: Any,
    ):
        """
        Initialise a ModelRelease with defensive handling for edge‑cases.

        - Empty strings for required fields raise ``ValueError``.
        - ``None`` collections default to empty ``dict``.
        - ``traffic_pct`` is clamped to the inclusive range [0, 100].
        - Negative ``n_features`` or ``seq_len`` raise ``ValueError``.
        - ``notes`` empty string is normalised to ``None``.
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non‑empty string")
        if not version or not version.strip():
            raise ValueError("version must be a non‑empty string")
        if not artifact_path or not artifact_path.strip():
            raise ValueError("artifact_path must be a non‑empty string")
        if not created_by or not created_by.strip():
            raise ValueError("created_by must be a non‑empty string")

        # Normalise optional collections
        model_params = model_params or {}
        training_config = training_config or {}
        train_metrics = train_metrics or {}
        live_metrics = live_metrics or {}

        # Clamp traffic percentage and validate range
        traffic_pct = max(0.0, min(float(traffic_pct or 0.0), 100.0))

        # Validate integer fields if provided
        if n_features is not None and n_features < 0:
            raise ValueError("n_features cannot be negative")
        if seq_len is not None and seq_len < 0:
            raise ValueError("seq_len cannot be negative")

        # Normalise notes
        notes = notes.strip() if notes and notes.strip() else None

        super().__init__(
            model_name=model_name.strip(),
            version=version.strip(),
            artifact_path=artifact_path.strip(),
            framework=framework.strip() if framework else "pytorch",
            n_features=n_features,
            seq_len=seq_len,
            model_params=model_params,
            training_config=training_config,
            train_metrics=train_metrics,
            live_metrics=live_metrics,
            status=status.strip() if status else "registered",
            traffic_pct=traffic_pct,
            notes=notes,
            created_by=created_by.strip(),
            **kwargs,
        )

    @validates("traffic_pct")
    def _validate_traffic_pct(self, key: str, value: Any) -> float:
        """
        Ensure traffic_pct stays within the allowed inclusive range.
        Handles ``None`` and off‑by‑one edge cases by clamping.
        """
        if value is None:
            return 0.0
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a numeric type, got {type(value)}")
        # Clamp to [0, 100] inclusive
        return max(0.0, min(val, 100.0))

    @validates("status")
    def _validate_status(self, key: str, value: Any) -> str:
        """
        Normalise status strings and guard against empty inputs.
        """
        if not value or not str(value).strip():
            raise ValueError("status cannot be empty")
        return str(value).strip().lower()

    @validates("model_params", "training_config", "train_metrics", "live_metrics")
    def _validate_json_blob(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Ensure JSON‑compatible columns are always dictionaries.
        ``None`` becomes an empty dict.
        """
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a dict, got {type(value)}")
        return value

    def __repr__(self) -> str:
        return (
            f"<ModelRelease id={self.id!r} model_name={self.model_name!r} "
            f"version={self.version!r} status={self.status!r}>"
        )