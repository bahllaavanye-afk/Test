import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import String, Boolean, ForeignKey, Numeric, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin


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

    def is_signal_allowed(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """
        Evaluate whether a trading signal meets tightened entry conditions.

        Entry criteria:
        1. Account must be active.
        2. Signal must contain required fields: ``price`` and ``volume``.
        3. ``price`` must be positive.
        4. ``volume`` must exceed the configured ``min_volume`` (default 0).
        5. Confirmation filter: price must be above the moving average (``ma``) if provided
           in ``market_data`` and ``ma_window`` is defined in ``extra_config``.

        Returns:
            bool: ``True`` if the signal passes all checks, ``False`` otherwise.
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
        """
        Determine if a position should be exited based on improved exit logic.

        Exit criteria:
        1. Stop‑loss: unrealized P&L falls below ``-stop_loss_pct`` of entry price.
        2. Take‑profit: unrealized P&L exceeds ``take_profit_pct`` of entry price.
        3. Trailing stop (optional): if ``trailing_stop_pct`` is set, exit when price drops
           a configured percentage from the highest price observed.

        Args:
            position: Dictionary containing at least ``entry_price`` and ``unrealized_pnl``.
            market_data: Dictionary containing current ``price`` and optionally ``high_price``.

        Returns:
            bool: ``True`` if the position meets any exit condition, ``False`` otherwise.
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
    __tablename__ = "account_snapshots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    total_equity: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    account: Mapped["Account"] = relationship("Account", back_populates="snapshots")