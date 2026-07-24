"""Bot model — declarative trading bot definitions."""
import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, JSON, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base
from app.models.base import TimestampMixin

# Canonical market types / desks a bot can trade. The column stays a free string for
# forward-compat, but this is the supported set the API advertises and the builder offers.
# equity/crypto/polymarket were the originals; options/macro/rates added in desk consolidation.
MARKET_TYPES: list[str] = ["equity", "crypto", "polymarket", "options", "macro", "rates"]


class Bot(Base, TimestampMixin):
    __tablename__ = "bots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    account_id: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(20), default="equity")  # see MARKET_TYPES

    trigger: Mapped[dict] = mapped_column(JSON, nullable=False)
    conditions: Mapped[list] = mapped_column(JSON, default=list)
    condition_logic: Mapped[str] = mapped_column(String(8), default="ALL")  # ALL | ANY
    action: Mapped[dict] = mapped_column(JSON, nullable=False)
    exit_rules: MMapped[list] = mapped_column(JSON, default=list)

    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Soft-delete / retire: archived bots are hidden from active lists, the desk
    # summary, and the scheduler, but their row + config + linked trades are preserved
    # so they can be restored or audited later.
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_signal: Mapped[str | None] = mapped_column(String(16), nullable=True)  # buy|sell|hold|alert
    last_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    template_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


# ----------------------------------------------------------------------
# Unit tests for edge‑case behavior
# ----------------------------------------------------------------------
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use an in‑memory SQLite database for fast isolated tests
_engine = create_engine("sqlite:///:memory:")
_Session = sessionmaker(bind=_engine)
Base.metadata.create_all(_engine)


def _create_bot_instance(**overrides):
    """Helper to create a Bot with required minimal fields."""
    defaults = {
        "user_id": "user-123",
        "name": "Test Bot",
        "symbol": "TEST",
        "trigger": {"type": "price"},
        "action": {"type": "order"},
    }
    defaults.update(overrides)
    return Bot(**defaults)


def test_bot_defaults_and_uuid():
    """Validate that defaults are applied and the generated id is a UUID string."""
    bot = _create_bot_instance()
    # UUID string should be 36 characters long and contain hyphens
    assert isinstance(bot.id, str)
    assert len(bot.id) == 36
    assert bot.id.count("-") == 4
    # Default market_type should be 'equity'
    assert bot.market_type == "equity"
    # Default mutable fields should be independent per instance
    bot2 = _create_bot_instance()
    assert bot.conditions is not bot2.conditions
    assert bot.exit_rules is not bot2.exit_rules


def test_bot_archived_flag_and_timestamp():
    """Edge case: archiving a bot should set is_archived and allow a nullable timestamp."""
    now = datetime.utcnow()
    bot = _create_bot_instance(is_archived=True, archived_at=now)
    # Verify flags and timestamp are stored correctly
    assert bot.is_archived is True
    assert bot.archived_at == now
    # Ensure that a bot without explicit archived_at remains None
    bot2 = _create_bot_instance(is_archived=True)
    assert bot2.archived_at is None


def test_bot_persistence_roundtrip():
    """Ensure that a Bot can be persisted and retrieved with all fields intact."""
    session = _Session()
    bot = _create_bot_instance(
        description="Edge case bot",
        condition_logic="ANY",
        conditions=[{"type": "volume", "threshold": 1000}],
        exit_rules=[{"type": "time", "limit": 60}],
    )
    session.add(bot)
    session.commit()
    retrieved = session.query(Bot).filter_by(id=bot.id).one()
    assert retrieved.id == bot.id
    assert retrieved.description == "Edge case bot"
    assert retrieved.condition_logic == "ANY"
    assert retrieved.conditions == [{"type": "volume", "threshold": 1000}]
    assert retrieved.exit_rules == [{"type": "time", "limit": 60}]
    session.close()