import uuid
from datetime import datetime, date
from typing import Dict, Optional

from sqlalchemy import String, Numeric, DateTime, Date, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ComparisonResult(Base):
    """
    SQLAlchemy model that stores the results of a back‑test comparison between a
    manual strategy and its machine‑learning‑enhanced counterpart.

    Each record captures performance metrics for the manual strategy, the ML‑enhanced
    strategy, and a benchmark (e.g., SPY).  Statistical significance of the
    difference is also recorded, together with the equity curves required for
    chart rendering.

    Attributes
    ----------
    id: str
        Primary key generated as a UUID string.
    strategy_name: str
        Human‑readable name of the strategy under test.
    symbol: str
        Ticker symbol the strategy was applied to.
    interval: str
        Data interval (e.g., ``"1d"``, ``"5m"``).
    start_date: date
        Inclusive start date of the back‑test period.
    end_date: date
        Inclusive end date of the back‑test period.
    manual_sharpe: Optional[float]
        Sharpe ratio of the manual strategy.
    manual_sortino: Optional[float]
        Sortino ratio of the manual strategy.
    manual_return: Optional[float]
        Total return of the manual strategy.
    manual_max_dd: Optional[float]
        Maximum drawdown of the manual strategy.
    manual_win_rate: Optional[float]
        Win‑rate (percentage of profitable trades) for the manual strategy.
    ml_sharpe: Optional[float]
        Sharpe ratio of the ML‑enhanced strategy.
    ml_sortino: Optional[float]
        Sortino ratio of the ML‑enhanced strategy.
    ml_return: Optional[float]
        Total return of the ML‑enhanced strategy.
    ml_max_dd: Optional[float]
        Maximum drawdown of the ML‑enhanced strategy.
    ml_win_rate: Optional[float]
        Win‑rate for the ML‑enhanced strategy.
    spy_sharpe: Optional[float]
        Sharpe ratio of the SPY benchmark over the same period.
    spy_return: Optional[float]
        Total return of the SPY benchmark.
    t_statistic: Optional[float]
        t‑statistic from a paired test comparing manual vs. ML performance.
    p_value: Optional[float]
        Two‑tailed p‑value associated with ``t_statistic``.
    is_significant: Optional[bool]
        Whether the result is statistically significant (``p_value`` below a
        predefined threshold).
    winner: Optional[str]
        Identifier of the winning approach: ``"manual"``, ``"ml"``, or ``"neither"``.
    equity_curves: Optional[Dict]
        JSON‑serialisable representation of the equity curves for both strategies.
    created_at: datetime
        Timestamp when the record was created (UTC).
    """

    __tablename__ = "comparison_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
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

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)