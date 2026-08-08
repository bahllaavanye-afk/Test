import uuid
from datetime import datetime
from functools import lru_cache
from typing import Any, List, Optional

import numpy as np
from sqlalchemy import DateTime, JSON, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

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
    metrics_history: Mapped[List[dict]] = mapped_column(JSON, default=list)  # [{epoch, loss, acc}, ...]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    def _extract_metric_values(self, metric_key: str) -> List[float]:
        """
        Extracts a list of numeric values for a given metric from ``metrics_history``.
        Returns an empty list if the metric is not present in any entry.
        """
        if not self.metrics_history:
            return []
        return [
            float(entry[metric_key])
            for entry in self.metrics_history
            if metric_key in entry and isinstance(entry[metric_key], (int, float))
        ]

    @lru_cache(maxsize=32)
    def average_metric(self, metric_key: str) -> Optional[float]:
        """
        Computes the average of a numeric metric across the stored ``metrics_history``.
        The result is cached to avoid repeated expensive calculations when the underlying
        ``metrics_history`` does not change.

        Parameters
        ----------
        metric_key: str
            The key of the metric to average (e.g., ``"accuracy"``, ``"loss"``).

        Returns
        -------
        Optional[float]
            The mean value of the metric, or ``None`` if the metric is absent.
        """
        values = self._extract_metric_values(metric_key)
        if not values:
            return None

        # Use NumPy for vectorized mean calculation when the list is large.
        # Fallback to pure Python for very small lists to avoid import overhead.
        if len(values) > 1000:
            return float(np.mean(values))
        return sum(values) / len(values)