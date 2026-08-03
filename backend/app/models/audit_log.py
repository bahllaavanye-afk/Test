import uuid
import logging
from datetime import datetime, timezone
from sqlalchemy import String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

logger = logging.getLogger(__name__)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(64))  # "order_submit", "order_cancel", "login", "key_add"
    resource_type: MMapped[str | None] = mapped_column(String(32))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(256))
    extra_data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def log_metrics(
        self,
        signal_count: int = 0,
        execution_time_ms: float = 0.0,
        pnl: float = 0.0,
    ) -> None:
        """
        Emit a structured INFO‑level log entry with key metrics.

        Parameters
        ----------
        signal_count: int
            Number of signals processed for this action.
        execution_time_ms: float
            Execution time in milliseconds.
        pnl: float
            Profit & loss associated with the action.
        """
        logger.info(
            "AuditLog entry created",
            extra={
                "audit_id": self.id,
                "user_id": self.user_id,
                "action": self.action,
                "resource_type": self.resource_type,
                "resource_id": self.resource_id,
                "signal_count": signal_count,
                "execution_time_ms": execution_time_ms,
                "pnl": pnl,
            },
        )