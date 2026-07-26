import uuid
from datetime import datetime, date
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
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


# ==============================
# Unit tests for edge cases
# ==============================
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def session():
    """Create an in‑memory SQLite session for isolated testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def test_backtestrun_dates_boundary(session):
    """Boundary test where start_date equals end_date."""
    today = date.today()
    run = BacktestRun(
        user_id="user123",
        strategy_name="test_strategy",
        symbol="AAPL",
        interval="1d",
        start_date=today,
        end_date=today,
        created_at=datetime.utcnow(),
    )
    session.add(run)
    session.commit()
    fetched = session.query(BacktestRun).filter_by(id=run.id).one()
    assert fetched.start_date == fetched.end_date == today

def test_backtestresult_relationship(session):
    """Ensure the one‑to‑one relationship between BacktestRun and BacktestResult works."""
    run = BacktestRun(
        user_id="user456",
        strategy_name="rel_test",
        symbol="GOOG",
        interval="1h",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 2),
        created_at=datetime.utcnow(),
    )
    result = BacktestResult(
        total_return=0.05,
        annualized_return=0.12,
        sharpe_ratio=1.5,
        run=run,
    )
    session.add_all([run, result])
    session.commit()
    fetched_result = session.query(BacktestResult).filter_by(id=result.id).one()
    assert fetched_result.run.id == run.id
    # The back‑reference from run should point to the same result
    assert fetched_result.run.result.id == fetched_result.id

def test_default_params_and_json_fields(session):
    """Validate default JSON fields and that optional JSON columns remain None when not set."""
    run = BacktestRun(
        user_id="user789",
        strategy_name="json_test",
        symbol="MSFT",
        interval="5m",
        start_date=date(2022, 6, 1),
        end_date=date(2022, 6, 30),
        created_at=datetime.utcnow(),
    )
    session.add(run)
    session.commit()
    fetched_run = session.query(BacktestRun).filter_by(id=run.id).one()
    assert fetched_run.params == {}
    # Create a result without explicit equity_curve/trades_log
    result = BacktestResult(run_id=fetched_run.id)
    session.add(result)
    session.commit()
    fetched_result = session.query(BacktestResult).filter_by(id=result.id).one()
    assert fetched_result.equity_curve is None
    assert fetched_result.trades_log is None