import uuid
from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


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

    # ML-enhanced strategy metrics
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
