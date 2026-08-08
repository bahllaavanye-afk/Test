import uuid
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import String, Boolean, ForeignKey, Numeric, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin


"""Models defining account structures and snapshots for the trading platform.

The module contains SQLAlchemy ORM models used throughout the system:

* :class:`Account` – Represents a brokerage account linked to a user, storing
  configuration, status, and relationships to orders, positions, etc.
* :class:`AccountSnapshot` – Periodic snapshot of an account's equity, cash,
  and unrealized P&L for historical analysis.
"""


class Account(Base, TimestampMixin):
    """SQLAlchemy model for a user's brokerage account.

    Attributes
    ----------
    id : Mapped[str]
        Primary key, generated as a UUID string.
    user_id : Mapped[str]
        Foreign key referencing the owning user.
    broker : Mapped[str]
        Broker identifier (e.g., ``alpaca``, ``tradestation``, ``binance``).
    label : Mapped[str]
        Human‑readable label for the account.
    mode : Mapped[str]
        Execution mode, either ``paper`` or ``live``.
    encrypted_key : Mapped[str | None]
        Encrypted API key; optional.
    encrypted_secret : Mapped[str | None]
        Encrypted API secret; optional.
    extra_config : Mapped[dict]
        JSON‑serialisable dictionary for account‑specific configuration such as
        volume thresholds or moving‑average windows.
    is_active : Mapped[bool]
        Flag indicating whether the account is currently active.

    Relationships
    -------------
    user : Mapped["User"]
        Back‑reference to the owning user.
    snapshots : Mapped[List["AccountSnapshot"]]
        Historical equity snapshots.
    orders : Mapped[List["Order"]]
        Orders placed against this account.
    positions : Mapped[List["Position"]]
        Open positions held by the account.
    strategies : Mapped[List["Strategy"]]
        Trading strategies attached to the account.
    """

    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    broker: Mapped[str] = mapped_column(String(50), nullable=False)  # alpaca|tradestation|binance|polymarket
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="paper")  # paper|live
    encrypted_key: Mapped[str | None] = mapped_column(String(1024))
    encrypted_secret: Mapped[str | None] = mapped_column(String(1024))
    extra_config: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped["User"] = relationship("User", back_populates="accounts")
    snapshots: Mapped[List["AccountSnapshot"]] = relationship("AccountSnapshot", back_populates="account")
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="account")
    positions: Mapped[List["Position"]] = relationship("Position", back_populates="account")
    strategies: Mapped[List["Strategy"]] = relationship("Strategy", back_populates="account")

    def is_signal_allowed(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Determine whether a trading signal satisfies entry constraints.

        Parameters
        ----------
        signal : Dict[str, Any]
            Signal payload expected to contain ``price`` and ``volume`` keys.
        market_data : Dict[str, Any]
            Current market context, optionally containing a moving average ``ma``.

        Returns
        -------
        bool
            ``True`` if the account is active and the signal meets all configured
            thresholds; otherwise ``False``.
        """
        if not self.is_active:
            return False

        price = signal.get("price")
        volume = signal.get("volume")
        if price is None or volume is None:
            return False
        if price <= 0:
            return False

        min_volume = self.extra_config.get("min_volume", 0)
        if volume < min_volume:
            return False

        ma_window = self.extra_config.get("ma_window")
        if ma_window is not None:
            ma = market_data.get("ma")
            if ma is None:
                return False
            if price <= ma:
                return False

        return True

    def should_exit_position(self, position: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Assess whether an open position should be closed based on exit rules.

        Parameters
        ----------
        position : Dict[str, Any]
            Must contain ``entry_price`` and ``unrealized_pnl`` entries.
        market_data : Dict[str, Any]
            Provides the current ``price`` and optionally ``high_price`` for trailing‑stop logic.

        Returns
        -------
        bool
            ``True`` if any stop‑loss, take‑profit, or trailing‑stop condition is met;
            otherwise ``False``.
        """
        entry_price = position.get("entry_price")
        unrealized_pnl = position.get("unrealized_pnl")
        if entry_price is None or unrealized_pnl is None:
            return False

        stop_loss_pct = self.extra_config.get("stop_loss_pct", 0.05)  # 5% default
        take_profit_pct = self.extra_config.get("take_profit_pct", 0.10)  # 10% default

        if unrealized_pnl <= -stop_loss_pct * entry_price:
            return True
        if unrealized_pnl >= take_profit_pct * entry_price:
            return True

        trailing_stop_pct = self.extra_config.get("trailing_stop_pct")
        if trailing_stop_pct is not None:
            high_price = market_data.get("high_price")
            current_price = market_data.get("price")
            if high_price is not None and current_price is not None:
                if (high_price - current_price) / high_price >= trailing_stop_pct:
                    return True

        return False


class AccountSnapshot(Base):
    """SQLAlchemy model capturing a point‑in‑time view of an account's financial state.

    Attributes
    ----------
    id : Mapped[str]
        Primary key, generated as a UUID string.
    account_id : Mapped[str]
        Foreign key linking to the ``Account``.
    ts : Mapped[datetime]
        Timestamp of the snapshot (timezone‑aware).
    total_equity : Mapped[float]
        Total equity value at the snapshot time.
    cash : Mapped[float]
        Cash balance.
    unrealized_pnl : Mapped[float]
        Unrealized profit and loss.
    raw_payload : Mapped[dict]
        Raw JSON payload received from the broker for reference.
    """

    __tablename__ = "account_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    total_equity: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    account: Mapped["Account"] = relationship("Account", back_populates="snapshots")