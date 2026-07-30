import uuid
import logging
from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin

logger = logging.getLogger(__name__)

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    accounts: Mapped[list["Account"]] = relationship("Account", back_populates="user", lazy="select")

    def log_metrics(self, signal_count: int, execution_time: float, pnl: float) -> None:
        """Log key performance metrics for the user.

        Args:
            signal_count: Number of signals generated.
            execution_time: Execution time in seconds.
            pnl: Profit and loss amount.
        """
        logger.info(
            "User metrics",
            extra={
                "user_id": self.id,
                "email": self.email,
                "signal_count": signal_count,
                "execution_time": execution_time,
                "pnl": pnl,
            },
        )