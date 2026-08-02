import uuid
import logging
from datetime import datetime, date
from sqlalchemy import String, Numeric, DateTime, Date, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

logger = logging.getLogger(__name__)


class ComparisonResult(Base):
    __tablename__ = "comparison_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Manual strategy metrics
    manual_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    manual_sortino: Mapped[float | None] = mapped_column(Numeric(8, 4))
    manual_return: Mapped[float | None] = mapped_column(Numeric(10, 4))
    manual_max_dd: Mapped[float | None] = mapped_column(Numeric(8, 4))
    manual_win_rate: Mapped[float | None] = mapped_column(Numeric(6, 4))

    # ML‑enhanced strategy metrics
    ml_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    ml_sortino: Mapped[float | None] = mapped_column(Numeric(8, 4))
    ml_return: Mapped[float | None] = mapped_column(Numeric(10, 4))
    ml_max_dd: Mapped[float | None] = mapped_column(Numeric(8, 4))
    ml_win_rate: Mapped[float | None] = mapped_column(Numeric(6, 4))

    # Benchmark metrics
    spy_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    spy_return: Mapped[float | None] = mapped_column(Numeric(10, 4))

    # Statistical significance
    t_statistic: Mapped[float | None] = mapped_column(Numeric(8, 4))
    p_value: Mapped[float | None] = mapped_column(Numeric(8, 6))
    is_significant: Mapped[bool | None] = mapped_column(Boolean)
    winner: Mapped[str | None] = mapped_column(String(8))  # manual|ml|neither

    # Full equity curves for chart rendering
    equity_curves: Mapped[dict | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def __init__(self, **kwargs):
        """
        Initialise the model and emit a structured log entry with key metrics.
        Expected keys in kwargs:
            - signal_count: int (optional)
            - execution_time: float (seconds, optional)
            - pnl: float (profit & loss, optional)
        """
        super().__init__(**kwargs)

        # Derive simple metrics if not explicitly provided
        signal_count = kwargs.get("signal_count")
        execution_time = kwargs.get("execution_time")
        pnl = kwargs.get("pnl")

        # Fallback calculations using existing fields
        if signal_count is None and isinstance(self.equity_curves, dict):
            signal_count = len(self.equity_curves.get("signals", []))

        if execution_time is None:
            # Approximate execution time as days between start and end
            try:
                delta = (self.end_date - self.start_date).days
                execution_time = delta * 86400.0  # convert days to seconds
            except Exception:
                execution_time = None

        if pnl is None:
            # Prefer manual_return, then ml_return, then spy_return
            pnl = (
                self.manual_return
                or self.ml_return
                or self.spy_return
            )

        log_payload = {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "interval": self.interval,
            "signal_count": signal_count,
            "execution_time_seconds": execution_time,
            "pnl": pnl,
        }

        logger.info("ComparisonResult created", extra=log_payload)