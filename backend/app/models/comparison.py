import uuid
import logging
from datetime import datetime, date

from sqlalchemy import String, Numeric, DateTime, Date, Boolean, JSON, event
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


def _log_comparison_result(mapper, connection, target: ComparisonResult):
    """Log key metrics of a ComparisonResult at INFO level."""
    # Prepare a structured dict with the most relevant metrics.
    log_payload = {
        "id": target.id,
        "strategy_name": target.strategy_name,
        "symbol": target.symbol,
        "interval": target.interval,
        "period": {
            "start": target.start_date.isoformat(),
            "end": target.end_date.isoformat(),
        },
        "manual_return": float(target.manual_return) if target.manual_return is not None else None,
        "ml_return": float(target.ml_return) if target.ml_return is not None else None,
        "pnl": (
            float(target.ml_return) - float(target.manual_return)
            if target.ml_return is not None and target.manual_return is not None
            else None
        ),
        "signal_count": len(target.equity_curves) if isinstance(target.equity_curves, dict) else None,
        "created_at": target.created_at.isoformat(),
    }
    logger.info("ComparisonResult persisted", extra=log_payload)


# Attach listeners to log after insert and after update.
event.listen(ComparisonResult, "after_insert", _log_comparison_result)
event.listen(ComparisonResult, "after_update", _log_comparison_result)