import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, ForeignKey, Numeric, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin

# Table names
TABLE_ACCOUNTS = "accounts"
TABLE_ACCOUNT_SNAPSHOTS = "account_snapshots"

# Foreign key references
FK_USERS_ID = "users.id"
FK_ACCOUNTS_ID = "accounts.id"

# Column length limits
BROKER_MAX_LENGTH = 50
LABEL_MAX_LENGTH = 100
MODE_MAX_LENGTH = 10
ENCRYPTED_MAX_LENGTH = 1024

# Default values
DEFAULT_MODE = "paper"
DEFAULT_ACTIVE = True

# Numeric precision/scale
NUMERIC_PRECISION = 18
NUMERIC_SCALE = 6


class Account(Base, TimestampMixin):
    __tablename__ = TABLE_ACCOUNTS

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(FK_USERS_ID, ondelete="CASCADE"),
        nullable=False,
    )
    broker: Mapped[str] = mapped_column(
        String(BROKER_MAX_LENGTH), nullable=False
    )  # alpaca|tradestation|binance|polymarket
    label: Mapped[str] = mapped_column(
        String(LABEL_MAX_LENGTH), nullable=False
    )
    mode: Mapped[str] = mapped_column(
        String(MODE_MAX_LENGTH), nullable=False, default=DEFAULT_MODE
    )  # paper|live
    encrypted_key: Mapped[str | None] = mapped_column(
        String(ENCRYPTED_MAX_LENGTH)
    )
    encrypted_secret: Mapped[str | None] = mapped_column(
        String(ENCRYPTED_MAX_LENGTH)
    )
    extra_config: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=DEFAULT_ACTIVE)

    user: Mapped["User"] = relationship("User", back_populates="accounts")
    snapshots: Mapped[list["AccountSnapshot"]] = relationship(
        "AccountSnapshot", back_populates="account"
    )
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="account")
    positions: Mapped[list["Position"]] = relationship(
        "Position", back_populates="account"
    )
    strategies: Mapped[list["Strategy"]] = relationship(
        "Strategy", back_populates="account"
    )


class AccountSnapshot(Base):
    __tablename__ = TABLE_ACCOUNT_SNAPSHOTS

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(FK_ACCOUNTS_ID, ondelete="CASCADE"),
        index=True,
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    total_equity: Mapped[float] = mapped_column(
        Numeric(NUMERIC_PRECISION, NUMERIC_SCALE), nullable=False
    )
    cash: Mapped[float] = mapped_column(
        Numeric(NUMERIC_PRECISION, NUMERIC_SCALE), nullable=False
    )
    unrealized_pnl: Mapped[float] = mapped_column(
        Numeric(NUMERIC_PRECISION, NUMERIC_SCALE), nullable=False
    )
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    account: Mapped["Account"] = relationship("Account", back_populates="snapshots")