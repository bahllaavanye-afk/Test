import uuid
from datetime import datetime, date
from typing import Dict, Optional

from sqlalchemy import String, Numeric, DateTime, Date, Boolean, JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _generate_uuid() -> str:
    """Generate a UUID string for primary keys."""
    return str(uuid.uuid4())


class ComparisonResult(Base):
    """
    SQLAlchemy model that stores the results of a back‑test comparison between a
    manual strategy and its machine‑learning‑enhanced counterpart.

    Each record captures performance metrics for the manual strategy, the ML‑enhanced
    strategy, and a benchmark (e.g., SPY).  Statistical significance of the
    difference is also recorded, together with the equity curves required for
    chart rendering.
    """

    __tablename__ = "comparison_results"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=_generate_uuid
    )
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Manual strategy metrics
    manual_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    manual_sortino: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    manual_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    manual_max_dd: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    manual_win_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))

    # ML‑enhanced strategy metrics
    ml_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ml_sortino: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ml_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    ml_max_dd: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ml_win_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))

    # Benchmark metrics
    spy_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    spy_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))

    # Statistical significance
    t_statistic: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    p_value: Mapped[Optional[float]] = mapped_column(Numeric(8, 6))
    is_significant: Mapped[Optional[bool]] = mapped_column(Boolean)
    winner: Mapped[Optional[str]] = mapped_column(String(8))  # manual|ml|neither

    # Full equity curves for chart rendering
    equity_curves: Mapped[Optional[Dict]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<ComparisonResult id={self.id!r} strategy={self.strategy_name!r} "
            f"symbol={self.symbol!r} interval={self.interval!r}>"
        )