import uuid
from datetime import datetime, date
from sqlalchemy import String, Numeric, DateTime, Date, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

# Constants
TABLE_NAME = "comparison_results"

STRATEGY_NAME_MAX_LENGTH = 64
SYMBOL_MAX_LENGTH = 32
INTERVAL_MAX_LENGTH = 8
WINNER_MAX_LENGTH = 8

NUMERIC_SHARPE_PRECISION = (8, 4)
NUMERIC_RETURN_PRECISION = (10, 4)
NUMERIC_WIN_RATE_PRECISION = (6, 4)
NUMERIC_PVALUE_PRECISION = (8, 6)

WINNER_OPTIONS = ("manual", "ml", "neither")


class ComparisonResult(Base):
    __tablename__ = TABLE_NAME

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_name: Mapped[str] = mapped_column(String(STRATEGY_NAME_MAX_LENGTH), nullable=False)
    symbol: Mapped[str] = mapped_column(String(SYMBOL_MAX_LENGTH), nullable=False)
    interval: Mapped[str] = mapped_column(String(INTERVAL_MAX_LENGTH), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Manual strategy metrics
    manual_sharpe: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_SHARPE_PRECISION))
    manual_sortino: MMapped[float | None] = mapped_column(Numeric(*NUMERIC_SHARPE_PRECISION))
    manual_return: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_RETURN_PRECISION))
    manual_max_dd: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_SHARPE_PRECISION))
    manual_win_rate: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_WIN_RATE_PRECISION))

    # ML-enhanced strategy metrics
    ml_sharpe: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_SHARPE_PRECISION))
    ml_sortino: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_SHARPE_PRECISION))
    ml_return: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_RETURN_PRECISION))
    ml_max_dd: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_SHARPE_PRECISION))
    ml_win_rate: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_WIN_RATE_PRECISION))

    # Benchmark metrics
    spy_sharpe: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_SHARPE_PRECISION))
    spy_return: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_RETURN_PRECISION))

    # Statistical significance
    t_statistic: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_SHARPE_PRECISION))
    p_value: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_PVALUE_PRECISION))
    is_significant: Mapped[bool | None] = mapped_column(Boolean)
    winner: Mapped[str | None] = mapped_column(String(WINNER_MAX_LENGTH))  # manual|ml|neither

    # Full equity curves for chart rendering
    equity_curves: Mapped[dict | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)