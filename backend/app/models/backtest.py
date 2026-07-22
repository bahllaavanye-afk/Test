import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


"""SQLAlchemy ORM models for backtesting runs and their results.

The `BacktestRun` model stores metadata about a backtest execution, while the
`BacktestResult` model captures the computed performance metrics and trade logs
produced by that execution. These models are used throughout the platform to
persist and retrieve backtest data.
"""


class BacktestRun(Base):
    """Represents a single backtest execution request.

    Attributes
    ----------
    id : str
        Primary key, generated as a UUID string.
    user_id : str
        Identifier of the user who submitted the backtest.
    strategy_name : str
        Name of the strategy to be backtested.
    symbol : str
        Trading symbol (e.g., ``AAPL``) for the backtest.
    interval : str
        Timeframe of the data (e.g., ``1h``).
    start_date : date
        Inclusive start date for the backtest period.
    end_date : date
        Inclusive end date for the backtest period.
    params : dict
        Strategy‑specific parameters supplied by the user.
    status : str
        Current processing state (``queued``, ``running``, ``done``, ``failed``).
    started_at : datetime | None
        Timestamp when the backtest started execution.
    completed_at : datetime | None
        Timestamp when the backtest finished execution.
    error_message : str | None
        Error details if the backtest failed.
    created_at : datetime
        Timestamp when the record was created.
    result : BacktestResult | None
        One‑to‑one relationship to the associated result record.
    """

    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    result: Mapped["BacktestResult | None"] = relationship("BacktestResult", back_populates="run", uselist=False)


class BacktestResult(Base):
    """Stores the performance metrics and trade details for a completed backtest.

    Attributes
    ----------
    id : str
        Primary key, generated as a UUID string.
    run_id : str
        Foreign key linking to the corresponding ``BacktestRun``.
    total_return : float | None
        Overall return of the strategy over the backtest period.
    annualized_return : float | None
        Annualized version of ``total_return``.
    sharpe_ratio : float | None
        Sharpe ratio of the strategy.
    sortino_ratio : float | None
        Sortino ratio of the strategy.
    calmar_ratio : float | None
        Calmar ratio of the strategy.
    max_drawdown : float | None
        Maximum observed drawdown.
    win_rate : float | None
        Proportion of winning trades.
    profit_factor : float | None
        Ratio of gross profit to gross loss.
    total_trades : int | None
        Number of trades executed.
    equity_curve : list | None
        Time‑series of equity values (e.g., ``[{'ts': ..., 'value': ...}, ...]``).
    trades_log : list | None
        Detailed log of individual trades (e.g., ``[{'entry': ..., 'exit': ..., 'pnl': ...}, ...]``).
    run : BacktestRun
        Back‑reference to the associated ``BacktestRun``.
    """

    __tablename__ = "backtest_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("backtest_runs.id", ondelete="CASCADE"), unique=True)
    total_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    annualized_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    calmar_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    max_drawdown: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    win_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    profit_factor: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    total_trades: Mapped[Optional[int]] = mapped_column(Integer)
    equity_curve: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)   # [{ts, value}, ...]
    trades_log: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)     # [{entry, exit, pnl}, ...]

    run: Mapped["BacktestRun"] = relationship("BacktestRun", back_populates="result")