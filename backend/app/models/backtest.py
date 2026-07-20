import uuid
import logging
from datetime import datetime, date
from sqlalchemy import (
    String,
    ForeignKey,
    Numeric,
    DateTime,
    Date,
    Integer,
    JSON,
    Text,
    event,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

logger = logging.getLogger(__name__)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    result: Mapped["BacktestResult | None"] = relationship(
        "BacktestResult", back_populates="run", uselist=False
    )

    def log_summary(self) -> None:
        """Log key back‑test metrics at INFO level."""
        if not self.result:
            logger.info(
                "BacktestRun %s completed with no result.", self.id, extra={"run_id": self.id}
            )
            return

        exec_time = None
        if self.started_at and self.completed_at:
            exec_time = (self.completed_at - self.started_at).total_seconds()

        logger.info(
            "BacktestRun completed",
            extra={
                "run_id": self.id,
                "strategy": self.strategy_name,
                "symbol": self.symbol,
                "interval": self.interval,
                "status": self.status,
                "total_trades": self.result.total_trades,
                "total_return": float(self.result.total_return or 0),
                "annualized_return": float(self.result.annualized_return or 0),
                "sharpe_ratio": float(self.result.sharpe_ratio or 0),
                "execution_seconds": exec_time,
            },
        )


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(
        String, ForeignKey("backtest_runs.id", ondelete="CASCADE"), unique=True
    )
    total_return: Mapped[float | None] = mapped_column(Numeric(10, 4))
    annualized_return: Mapped[float | None] = mapped_column(Numeric(10, 4))
    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    sortino_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    calmar_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(8, 4))
    win_rate: Mapped[float | None] = mapped_column(Numeric(6, 4))
    profit_factor: Mapped[float | None] = mapped_column(Numeric(8, 4))
    total_trades: Mapped[int | None] = mapped_column(Integer)
    equity_curve: Mapped[list | None] = mapped_column(JSON)  # [{ts, value}, ...]
    trades_log: Mapped[list | None] = mapped_column(JSON)  # [{entry, exit, pnl}, ...]

    run: Mapped["BacktestRun"] = relationship("BacktestRun", back_populates="result")


def _after_backtest_run_update(mapper, connection, target: BacktestRun):
    """SQLAlchemy event listener to log when a run finishes."""
    if target.status in {"done", "failed"}:
        target.log_summary()


event.listen(BacktestRun, "after_update", _after_backtest_run_update)