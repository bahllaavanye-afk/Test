import uuid
from datetime import datetime
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Integer, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_account_closed", "account_id", "closed_at"),
        Index("ix_trades_strategy_closed", "strategy_id", "closed_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"), index=True)
    strategy_id: Mapped[str | None] = mapped_column(String, ForeignKey("strategies.id", ondelete="SET NULL"), index=True)
    # Denormalized for fast attribution queries (avoids JOIN to strategies table)
    strategy_name: Mapped[str | None] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    exit_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    fees: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    hold_seconds: Mapped[int | None] = mapped_column(Integer)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)


# ==============================
# Unit Tests for Trade Model
# ==============================
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class TestTradeModel(unittest.TestCase):
    def setUp(self):
        # Use an in‑memory SQLite database for isolated testing
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def test_default_values(self):
        """Verify defaults (fees, raw_payload) and UUID generation."""
        session = self.Session()
        now = datetime.utcnow()
        trade = Trade(
            account_id="acc1",
            strategy_id="strat1",
            strategy_name="TestStrategy",
            symbol="AAPL",
            side="BUY",
            entry_price=100.0,
            exit_price=105.0,
            quantity=10.0,
            realized_pnl=50.0,
            opened_at=now,
            closed_at=now,
        )
        session.add(trade)
        session.commit()
        fetched = session.query(Trade).first()
        # UUID format
        self.assertIsNotNone(fetched.id)
        uuid_obj = uuid.UUID(fetched.id)
        self.assertEqual(str(uuid_obj), fetched.id)
        # Default fields
        self.assertEqual(fetched.fees, 0)
        self.assertIsInstance(fetched.raw_payload, dict)
        self.assertEqual(fetched.raw_payload, {})

    def test_boundary_numeric_values(self):
        """Edge case: zero and negative numeric values, and explicit hold_seconds."""
        session = self.Session()
        trade = Trade(
            account_id="acc2",
            strategy_id=None,
            strategy_name=None,
            symbol="TSLA",
            side="SELL",
            entry_price=0.0,
            exit_price=-0.0,
            quantity=0.0,
            realized_pnl=-0.0,
            opened_at=datetime(1970, 1, 1),
            closed_at=datetime(1970, 1, 1),
            hold_seconds=0,
            fees=0.0,
            raw_payload={"key": "value"},
        )
        session.add(trade)
        session.commit()
        fetched = session.query(Trade).filter_by(symbol="TSLA").first()
        # Numeric columns are stored as Decimal; compare via float conversion
        self.assertEqual(float(fetched.entry_price), 0.0)
        self.assertEqual(float(fetched.exit_price), -0.0)
        self.assertEqual(float(fetched.quantity), 0.0)
        self.assertEqual(float(fetched.realized_pnl), -0.0)
        self.assertEqual(fetched.hold_seconds, 0)
        self.assertEqual(fetched.raw_payload, {"key": "value"})

    def test_missing_optional_fields(self):
        """Ensure optional fields can be omitted and default correctly."""
        session = self.Session()
        now = datetime.utcnow()
        trade = Trade(
            account_id="acc3",
            strategy_id=None,
            strategy_name=None,
            symbol="MSFT",
            side="BUY",
            entry_price=200.0,
            exit_price=210.0,
            quantity=5.0,
            realized_pnl=50.0,
            opened_at=now,
            closed_at=now,
            # hold_seconds and raw_payload omitted intentionally
        )
        session.add(trade)
        session.commit()
        fetched = session.query(Trade).filter_by(symbol="MSFT").first()
        self.assertIsNone(fetched.hold_seconds)
        self.assertEqual(fetched.raw_payload, {})


if __name__ == "__main__":
    unittest.main()