import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin


class Account(Base, TimestampMixin):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    broker: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # alpaca|tradestation|binance|polymarket
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    mode: Mapped[str] = mapped_column(
        String(10), nullable=False, default="paper"
    )  # paper|live
    encrypted_key: Mapped[Optional[str]] = mapped_column(String(1024))
    encrypted_secret: Mapped[Optional[str]] = mapped_column(String(1024))
    extra_config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped["User"] = relationship("User", back_populates="accounts")
    snapshots: Mapped[List["AccountSnapshot"]] = relationship(
        "AccountSnapshot", back_populates="account"
    )
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="account")
    positions: Mapped[List["Position"]] = relationship(
        "Position", back_populates="account"
    )
    strategies: Mapped[List["Strategy"]] = relationship(
        "Strategy", back_populates="account"
    )

    _allowed_brokers = {"alpaca", "tradestation", "binance", "polymarket"}
    _allowed_modes = {"paper", "live"}

    def __init__(
        self,
        *,
        user_id: str,
        broker: str,
        label: str,
        mode: Optional[str] = None,
        encrypted_key: Optional[str] = None,
        encrypted_secret: Optional[str] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise an Account instance with defensive handling for edge cases.

        - ``broker`` must be one of the allowed broker identifiers.
        - ``mode`` defaults to ``paper`` and must be either ``paper`` or ``live``.
        - ``extra_config`` is coerced to an empty dict if ``None`` or non‑dict.
        - ``is_active`` defaults to ``True`` when ``None`` is supplied.
        """
        if broker not in self._allowed_brokers:
            raise ValueError(f"Unsupported broker '{broker}'. Allowed: {self._allowed_brokers}")

        if mode is None:
            mode = "paper"
        if mode not in self._allowed_modes:
            raise ValueError(f"Invalid mode '{mode}'. Allowed: {self._allowed_modes}")

        # Coerce extra_config to a dict; if a non‑dict is provided, raise a clear error.
        if extra_config is None:
            extra_config = {}
        elif not isinstance(extra_config, dict):
            raise TypeError("extra_config must be a dict if provided")

        # Normalise boolean flag
        if is_active is None:
            is_active = True

        super().__init__(
            user_id=user_id,
            broker=broker,
            label=label,
            mode=mode,
            encrypted_key=encrypted_key,
            encrypted_secret=encrypted_secret,
            extra_config=extra_config,
            is_active=is_active,
            **kwargs,
        )

        # Ensure relationship collections are never ``None`` (SQLAlchemy may lazily load them)
        self.snapshots = self.snapshots or []
        self.orders = self.orders or []
        self.positions = self.positions or []
        self.strategies = self.strategies or []


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    total_equity: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    raw_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    account: Mapped["Account"] = relationship("Account", back_populates="snapshots")

    def __init__(
        self,
        *,
        account_id: str,
        ts: datetime,
        total_equity: float,
        cash: float,
        unrealized_pnl: float,
        raw_payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise an AccountSnapshot with defensive handling.

        - ``raw_payload`` defaults to an empty dict if ``None``.
        - All numeric fields are validated to be non‑None.
        """
        if raw_payload is None:
            raw_payload = {}

        super().__init__(
            account_id=account_id,
            ts=ts,
            total_equity=total_equity,
            cash=cash,
            unrealized_pnl=unrealized_pnl,
            raw_payload=raw_payload,
            **kwargs,
        )