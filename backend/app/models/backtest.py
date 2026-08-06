import uuid
from datetime import datetime, date, timezone
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, JSON, Text, create_engine
from sqlalchemy.orm import Mapped, mapped_column, relationship, sessionmaker, validates
from app.database import Base


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

    @validates('user_id', 'strategy_name', 'symbol', 'interval')
    def _validate_non_empty_string(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non‑empty string")
        return value

    @validates('start_date', 'end_date')
    def _validate_date(self, key: str, value: date) -> date:
        if not isinstance(value, date):
            raise ValueError(f"{key} must be a datetime.date instance")
        return value

    @validates('created_at')
    def _validate_created_at(self, key: str, value: datetime) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError("created_at must be a datetime instance")
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone‑aware")
        return value

    @validates('start_date', 'end_date')
    def _validate_date_order(self, key: str, value: date) -> date:
        # This validator runs for each date field; order check is performed after both are set
        # Defer ordering logic to a separate check on attribute set
        return value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure start_date is not after end_date
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be later than end_date")


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

    @validates('total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
               'calmar_ratio', 'max_drawdown', 'win_rate', 'profit_factor')
    def _validate_float_or_none(self, key: str, value):
        if value is not None and not isinstance(value, (float, int)):
            raise ValueError(f"{key} must be a float, int, or None")
        return float(value) if isinstance(value, (float, int)) else None

    @validates('total_trades')
    def _validate_int_or_none(self, key: str, value):
        if value is not None and not isinstance(value, int):
            raise ValueError(f"{key} must be an int or None")
        return value

    @validates('equity_curve', 'trades_log')
    def _validate_list_or_none(self, key: str, value):
        if value is not None and not isinstance(value, list):
            raise ValueError(f"{key} must be a list or None")
        return value


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