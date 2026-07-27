"""SQLAlchemy model for storing comparison results between manual and ML‑enhanced trading strategies.

This model captures performance metrics for both strategies, benchmark data, statistical
significance tests, and the full equity curves needed for chart rendering. It is used
by the backend to persist results of back‑testing runs and to serve data to the UI.
"""

import uuid
from datetime import datetime, date
from typing import Optional, Dict, Any

from sqlalchemy import String, Numeric, DateTime, Date, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ComparisonResult(Base):
    """ORM model representing a single comparison between manual and ML‑enhanced strategies.

    Attributes
    ----------
    id : Mapped[str]
        Primary key; generated as a UUID string.
    strategy_name : Mapped[str]
        Name of the strategy under comparison.
    symbol : Mapped[str]
        Trading symbol (e.g., ``AAPL``).
    interval : Mapped[str]
        Time‑frame interval used for the back‑test (e.g., ``1d``).
    start_date : Mapped[date]
        Inclusive start date of the back‑test period.
    end_date : Mapped[date]
        Inclusive end date of the back‑test period.
    manual_sharpe, manual_sortino, manual_return, manual_max_dd, manual_win_rate :
        Performance metrics for the manually‑executed strategy.
    ml_sharpe, ml_sortino, ml_return, ml_max_dd, ml_win_rate :
        Performance metrics for the ML‑enhanced strategy.
    spy_sharpe, spy_return :
        Benchmark metrics using the SPY ETF.
    t_statistic, p_value, is_significant, winner :
        Results of statistical significance testing.
    equity_curves :
        Serialized equity curve data for both strategies, stored as JSON.
    created_at :
        Timestamp when the record was created.
    """

    __tablename__ = "comparison_results"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
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

    # ML-enhanced strategy metrics
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
    equity_curves: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    def __repr__(self) -> str:
        """Return a concise string representation useful for debugging."""
        return (
            f"<ComparisonResult id={self.id!r} strategy={self.strategy_name!r} "
            f"symbol={self.symbol!r} interval={self.interval!r}>"
        )