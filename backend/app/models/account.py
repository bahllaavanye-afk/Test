import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, ForeignKey, Numeric, DateTime, JSON, create_engine
from sqlalchemy.orm import Mapped, mapped_column, relationship, sessionmaker
from app.database import Base
from app.models.base import TimestampMixin

# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------


class Account(Base, TimestampMixin):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    broker: Mapped[str] = mapped_column(String(50), nullable=False)  # alpaca|tradestation|binance|polymarket
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="paper")  # paper|live
    encrypted_key: Mapped[str | None] = mapped_column(String(1024))
    encrypted_secret: Mapped[str | None] = mapped_column(String(1024))
    extra_config: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped["User"] = relationship("User", back_populates="accounts")
    snapshots: Mapped[list["AccountSnapshot"]] = relationship("AccountSnapshot", back_populates="account")
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="account")
    positions: Mapped[list["Position"]] = relationship("Position", back_populates="account")
    strategies: Mapped[list["Strategy"]] = relationship("Strategy", back_populates="account")


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    total_equity: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    account: Mapped["Account"] = relationship("Account", back_populates="snapshots")


# ----------------------------------------------------------------------
# Unit Tests (edge cases)
# ----------------------------------------------------------------------
import pytest

# Create an in‑memory SQLite engine for isolated testing
_TEST_ENGINE = create_engine("sqlite:///:memory:", echo=False)
_TestSession = sessionmaker(bind=_TEST_ENGINE)


@pytest.fixture(scope="function")
def session():
    """Provides a fresh SQLAlchemy session with tables created."""
    Base.metadata.create_all(_TEST_ENGINE)
    sess = _TestSession()
    yield sess
    sess.close()
    Base.metadata.drop_all(_TEST_ENGINE)


def test_account_defaults(session):
    """Boundary test: verify default field values and that mutable defaults are independent."""
    acct = Account(
        user_id="test_user",
        broker="alpaca",
        label="Test Account",
        encrypted_key=None,
        encrypted_secret=None,
    )
    session.add(acct)
    session.commit()

    # Refresh to ensure defaults are persisted
    session.refresh(acct)

    assert acct.mode == "paper", "Default mode should be 'paper'"
    assert acct.is_active is True, "Default is_active should be True"
    assert acct.extra_config == {}, "Default extra_config should be an empty dict"
    # Ensure each instance gets its own dict (no shared mutable default)
    acct2 = Account(
        user_id="test_user2",
        broker="binance",
        label="Second Account",
        encrypted_key=None,
        encrypted_secret=None,
    )
    session.add(acct2)
    session.commit()
    session.refresh(acct2)
    acct2.extra_config["key"] = "value"
    assert acct.extra_config == {}, "extra_config dict should be independent per instance"


def test_account_id_uuid_format(session):
    """Boundary test: ensure generated IDs conform to UUID4 string format (36 characters with hyphens)."""
    acct = Account(
        user_id="uuid_user",
        broker="tradestation",
        label="UUID Test",
        encrypted_key=None,
        encrypted_secret=None,
    )
    session.add(acct)
    session.commit()
    session.refresh(acct)

    uuid_str = acct.id
    # UUID4 format: 8-4-4-4-12 hex characters
    parts = uuid_str.split("-")
    assert len(parts) == 5, "UUID should contain 5 hyphen-separated parts"
    assert all(len(p) == expected for p, expected in zip(parts, [8, 4, 4, 4, 12])), "Each part length must match UUID4 spec"
    # Validate that the string can be parsed by uuid.UUID
    try:
        parsed = uuid.UUID(uuid_str, version=4)
    except ValueError:
        pytest.fail("Account.id is not a valid UUID4 string")
    assert str(parsed) == uuid_str.lower(), "Parsed UUID should match original string (case‑insensitive)"


def test_account_snapshot_creation(session):
    """Boundary test: create a snapshot with extreme numeric values to ensure Numeric column handling."""
    acct = Account(
        user_id="snap_user",
        broker="polymarket",
        label="Snapshot Test",
        encrypted_key=None,
        encrypted_secret=None,
    )
    session.add(acct)
    session.commit()
    session.refresh(acct)

    # Use values at the precision limit of Numeric(18,6)
    max_value = 999999999999.999999  # 12 digits before decimal, 6 after
    snap = AccountSnapshot(
        account_id=acct.id,
        ts=datetime.utcnow(),
        total_equity=max_value,
        cash=max_value,
        unrealized_pnl=-max_value,
        raw_payload={"detail": "extreme values"},
    )
    session.add(snap)
    session.commit()
    session.refresh(snap)

    assert float(snap.total_equity) == max_value, "total_equity should retain the max allowed precision"
    assert float(snap.cash) == max_value, "cash should retain the max allowed precision"
    assert float(snap.unrealized_pnl) == -max_value, "unrealized_pnl should retain negative max precision"
    assert snap.raw_payload == {"detail": "extreme values"}, "raw_payload should be stored unchanged"