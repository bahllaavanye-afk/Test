import uuid
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, JSON, Text, create_engine
from sqlalchemy.orm import Mapped, mapped_column, relationship, sessionmaker, validates

from app.database import Base


def _ensure_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non‑empty string")
    return value


def _ensure_date(value: Any, field_name: str) -> date:
    if not isinstance(value, date):
        raise ValueError(f"{field_name} must be a datetime.date instance")
    return value


def _ensure_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime.datetime instance")
    return value


def _ensure_dict(value: Any, field_name: str) -> Dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dict")
    return value


def _ensure_numeric(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number or None")
    return float(value)


def _ensure_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an int or None")
    return value


def _ensure_list(value: Any, field_name: str) -> List:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


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

    result: Mapped["BacktestResult | None"] = relationship("BacktestResult", back_populates="run", uselist=False)

    def __init__(
        self,
        user_id: str,
        strategy_name: str,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
        created_at: datetime,
        params: Optional[Dict] = None,
        status: str = "queued",
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.user_id = _ensure_str(user_id, "user_id")
        self.strategy_name = _ensure_str(strategy_name, "strategy_name")
        self.symbol = _ensure_str(symbol, "symbol")
        self.interval = _ensure_str(interval, "interval")
        self.start_date = _ensure_date(start_date, "start_date")
        self.end_date = _ensure_date(end_date, "end_date")
        self.created_at = _ensure_datetime(created_at, "created_at")
        self.params = _ensure_dict(params if params is not None else {}, "params")
        self.status = _ensure_str(status, "status")
        if started_at is not None:
            self.started_at = _ensure_datetime(started_at, "started_at")
        if completed_at is not None:
            self.completed_at = _ensure_datetime(completed_at, "completed_at")
        if error_message is not None:
            self.error_message = _ensure_str(error_message, "error_message")
        super().__init__(**kwargs)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("backtest_runs.id", ondelete="CASCADE"), unique=True)
    total_return: Mapped[float | None] = mapped_column(Numeric(10, 4))
    annualized_return: Mapped[float | None] = mapped_column(Numeric(10, 4))
    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    sortino_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    calmar_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(8, 4))
    win_rate: Mapped[float | None] = mapped_column(Numeric(6, 4))
    profit_factor: Mapped[float | None] = mapped_column(Numeric(8, 4))
    total_trades: Mapped[int | None] = mapped_column(Integer)
    equity_curve: Mapped[list | None] = mapped_column(JSON)   # [{ts, value}, ...]
    trades_log: Mapped[list | None] = mapped_column(JSON)     # [{entry, exit, pnl}, ...]

    run: Mapped["BacktestRun"] = relationship("BacktestRun", back_populates="result")

    def __init__(
        self,
        run_id: str,
        total_return: Optional[Union[int, float]] = None,
        annualized_return: Optional[Union[int, float]] = None,
        sharpe_ratio: Optional[Union[int, float]] = None,
        sortino_ratio: Optional[Union[int, float]] = None,
        calmar_ratio: Optional[Union[int, float]] = None,
        max_drawdown: Optional[Union[int, float]] = None,
        win_rate: Optional[Union[int, float]] = None,
        profit_factor: Optional[Union[int, float]] = None,
        total_trades: Optional[int] = None,
        equity_curve: Optional[List] = None,
        trades_log: Optional[List] = None,
        **kwargs: Any,
    ) -> None:
        self.run_id = _ensure_str(run_id, "run_id")
        self.total_return = _ensure_numeric(total_return, "total_return")
        self.annualized_return = _ensure_numeric(annualized_return, "annualized_return")
        self.sharpe_ratio = _ensure_numeric(sharpe_ratio, "sharpe_ratio")
        self.sortino_ratio = _ensure_numeric(sortino_ratio, "sortino_ratio")
        self.calmar_ratio = _ensure_numeric(calmar_ratio, "calmar_ratio")
        self.max_drawdown = _ensure_numeric(max_drawdown, "max_drawdown")
        self.win_rate = _ensure_numeric(win_rate, "win_rate")
        self.profit_factor = _ensure_numeric(profit_factor, "profit_factor")
        self.total_trades = _ensure_int(total_trades, "total_trades")
        self.equity_curve = _ensure_list(equity_curve if equity_curve is not None else [], "equity_curve")
        self.trades_log = _ensure_list(trades_log if trades_log is not None else [], "trades_log")
        super().__init__(**kwargs)


# ----------------------------------------------------------------------
# Unit tests for edge / boundary conditions
# ----------------------------------------------------------------------
import unittest


class TestBacktestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # In‑memory SQLite database for isolated testing
        engine = create_engine("sqlite:///:memory:", echo=False, future=True)
        Base.metadata.create_all(engine)
        cls.Session = sessionmaker(bind=engine, future=True)

    def test_default_uuid_format(self):
        """Ensure the autogenerated UUID string has the expected length and format."""
        session = self.Session()
        run = BacktestRun(
            user_id="user-123",
            strategy_name="mean_rev_20_1.5",
            symbol="AAPL",
            interval="1d",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            created_at=datetime.now(timezone.utc),
        )
        session.add(run)
        session.commit()
        # UUID string should be 36 characters (including hyphens)
        self.assertIsInstance(run.id, str)
        self.assertEqual(len(run.id), 36)
        session.close()

    def test_start_equals_end_date_boundary(self):
        """A backtest where start_date == end_date should be allowed (zero‑length period)."""
        session = self.Session()
        same_day = date(2024, 5, 17)
        run = BacktestRun(
            user_id="user-456",
            strategy_name="mean_rev_20_2",
            symbol="MSFT",
            interval="1h",
            start_date=same_day,
            end_date=same_day,
            created_at=datetime.now(timezone.utc),
        )
        session.add(run)
        session.commit()
        fetched = session.get(BacktestRun, run.id)
        self.assertEqual(fetched.start_date, fetched.end_date)
        session.close()

    def test_backtest_result_null_numeric_fields(self):
        """Numeric result fields should accept None without raising errors."""
        session = self.Session()
        run = BacktestRun(
            user_id="user-789",
            strategy_name="mean_rev_20_1.5",
            symbol="GOOG",
            interval="5m",
            start_date=date(2022, 1, 1),
            end_date=date(2022, 1, 31),
            created_at=datetime.now(timezone.utc),
        )
        session.add(run)
        session.flush()  # obtain run.id without committing

        result = BacktestResult(
            run_id=run.id,
            total_return=None,
            annualized_return=None,
            sharpe_ratio=None,
            sortino_ratio=None,
            calmar_ratio=None,
            max_drawdown=None,
            win_rate=None,
            profit_factor=None,
            total_trades=None,
            equity_curve=[],
            trades_log=[],
        )
        session.add(result)
        session.commit()

        fetched_result = session.get(BacktestResult, result.id)
        self.assertIsNone(fetched_result.total_return)
        self.assertIsInstance(fetched_result.equity_curve, list)
        self.assertEqual(fetched_result.equity_curve, [])
        session.close()


if __name__ == "__main__":
    unittest.main()