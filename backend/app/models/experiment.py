import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    val_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    val_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    test_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    metrics_history: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON, default=list
    )  # [{epoch, loss, acc}, ...]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    _VALID_STATUSES = {"queued", "running", "done", "failed"}

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        status: str = "queued",
        val_accuracy: Optional[float] = None,
        val_sharpe: Optional[float] = None,
        test_sharpe: Optional[float] = None,
        artifact_path: Optional[str] = None,
        metrics_history: Optional[List[Dict[str, Any]]] = None,
        created_at: Optional[datetime] = None,
    ) -> None:
        self._validate_name(name)
        self._validate_config(config)
        self._validate_status(status)
        self._validate_float(val_accuracy, "val_accuracy")
        self._validate_float(val_sharpe, "val_sharpe")
        self._validate_float(test_sharpe, "test_sharpe")
        self._validate_artifact_path(artifact_path)
        self._validate_metrics_history(metrics_history)

        self.name = name
        self.config = config
        self.status = status
        self.val_accuracy = val_accuracy
        self.val_sharpe = val_sharpe
        self.test_sharpe = test_sharpe
        self.artifact_path = artifact_path
        self.metrics_history = metrics_history if metrics_history is not None else []
        self.created_at = created_at or datetime.utcnow()

    @staticmethod
    def _validate_name(name: Any) -> None:
        if not isinstance(name, str):
            raise ValueError("Experiment name must be a string.")
        if not name.strip():
            raise ValueError("Experiment name cannot be empty or whitespace.")
        if len(name) > 128:
            raise ValueError("Experiment name must not exceed 128 characters.")

    @staticmethod
    def _validate_config(config: Any) -> None:
        if not isinstance(config, dict):
            raise ValueError("Experiment config must be a dictionary.")
        # Further structural validation can be added here as needed.

    @classmethod
    def _validate_status(cls, status: Any) -> None:
        if not isinstance(status, str):
            raise ValueError("Experiment status must be a string.")
        if status not in cls._VALID_STATUSES:
            raise ValueError(
                f"Experiment status must be one of {sorted(cls._VALID_STATUSES)}."
            )

    @staticmethod
    def _validate_float(value: Any, field_name: str) -> None:
        if value is None:
            return
        if not isinstance(value, (float, int)):
            raise ValueError(f"{field_name} must be a numeric type.")
        if isinstance(value, float) and (value != value):  # NaN check
            raise ValueError(f"{field_name} cannot be NaN.")

    @staticmethod
    def _validate_artifact_path(path: Any) -> None:
        if path is None:
            return
        if not isinstance(path, str):
            raise ValueError("artifact_path must be a string if provided.")
        if len(path) > 512:
            raise ValueError("artifact_path must not exceed 512 characters.")

    @staticmethod
    def _validate_metrics_history(history: Any) -> None:
        if history is None:
            return
        if not isinstance(history, list):
            raise ValueError("metrics_history must be a list of dictionaries.")
        for idx, entry in enumerate(history):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"metrics_history entry at index {idx} must be a dictionary."
                )
            # Optional: enforce expected keys such as 'epoch', 'loss', 'acc'
            # Example:
            # required_keys = {"epoch", "loss", "acc"}
            # missing = required_keys - entry.keys()
            # if missing:
            #     raise ValueError(
            #         f"metrics_history entry at index {idx} is missing keys: {missing}"
            #     )