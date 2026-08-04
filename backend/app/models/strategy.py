import uuid
import logging
from sqlalchemy import String, ForeignKey, Boolean, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin

logger = logging.getLogger(__name__)

class Strategy(Base, TimestampMixin):
    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str | None] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)          # e.g. 'pairs_trading'
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    market_type: Mapped[str] = mapped_column(String(16), nullable=False)   # equity|crypto|polymarket
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False) # manual|ml_enhanced
    risk_bucket: Mapped[str] = mapped_column(String(16), nullable=False)   # arbitrage|directional
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    symbols: Mapped[list] = mapped_column(JSON, default=list)              # tracked symbols
    tick_interval_seconds: Mapped[float] = mapped_column(Float, default=60.0)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.60)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    account: Mapped["Account"] = relationship("Account", back_populates="strategies")

    def log_metrics(self, signal_count: int, execution_time_seconds: float, pnl: float) -> None:
        """
        Log key performance metrics for the strategy.

        Parameters
        ----------
        signal_count : int
            Number of signals generated/executed in the current run.
        execution_time_seconds : float
            Execution time for the run in seconds.
        pnl : float
            Profit and loss realized for the run.
        """
        logger.info(
            "Strategy metrics",
            extra={
                "strategy_id": self.id,
                "name": self.name,
                "signal_count": signal_count,
                "execution_time_seconds": execution_time_seconds,
                "pnl": pnl,
            },
        )