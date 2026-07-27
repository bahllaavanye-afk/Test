import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.database import Base


class Trade(Base):
    """
    ORM model representing a trade execution.

    The model now includes helper methods to assess signal quality
    and exit conditions, providing tighter entry criteria and
    clearer exit logic without altering existing persistence behavior.
    """

    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_account_closed", "account_id", "closed_at"),
        Index("ix_trades_strategy_closed", "strategy_id", "closed_at"),
    )

    # ----------------------------------------------------------------------
    # Columns
    # ----------------------------------------------------------------------
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    account_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        index=True,
    )
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("strategies.id", ondelete="SET NULL"),
        index=True,
    )
    # Denormalized for fast attribution queries (avoids JOIN to strategies table)
    strategy_name: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    exit_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    fees: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    closed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    hold_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    raw_payload: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------
    @validates("side")
    def _validate_side(self, key: str, value: str) -> str:
        """Ensure side is either 'buy' or 'sell' (case‑insensitive)."""
        normalized = value.lower()
        if normalized not in {"buy", "sell"}:
            raise ValueError(f"Invalid side '{value}'. Must be 'buy' or 'sell'.")
        return normalized

    # ----------------------------------------------------------------------
    # Strategy‑level helpers
    # ----------------------------------------------------------------------
    DEFAULT_ENTRY_THRESHOLD: float = 0.55
    DEFAULT_VOLUME_THRESHOLD: float = 1.0  # Relative volume spike

    def _get_signal_strength(self) -> Optional[float]:
        """Extract signal strength from the payload if present."""
        return self.raw_payload.get("signal_strength")

    def _get_volume_spike(self) -> Optional[float]:
        """Extract volume spike factor from the payload if present."""
        return self.raw_payload.get("volume_spike")

    def entry_is_confident(self) -> bool:
        """
        Determine whether the entry signal meets tightened criteria.

        Criteria:
        * Signal strength must exceed a dynamic threshold (default 0.55).
        * Volume spike must be above a minimum relative level.
        """
        strength = self._get_signal_strength()
        volume = self._get_volume_spike()
        if strength is None or volume is None:
            return False

        threshold = float(
            self.raw_payload.get("entry_threshold", self.DEFAULT_ENTRY_THRESHOLD)
        )
        return strength > threshold and volume > self.DEFAULT_VOLUME_THRESHOLD

    def exit_is_triggered(self) -> bool:
        """
        Evaluate exit conditions using payload parameters.

        The method checks for:
        * Stop‑loss breach.
        * Take‑profit achievement.
        * Optional trailing‑stop condition.
        Returns True if any condition is satisfied.
        """
        stop_loss = self.raw_payload.get("stop_loss")
        take_profit = self.raw_payload.get("take_profit")
        trailing_stop = self.raw_payload.get("trailing_stop")

        # Compare against the actual exit price
        price = float(self.exit_price)

        if stop_loss is not None and ((self.side == "buy" and price <= stop_loss) or
                                      (self.side == "sell" and price >= stop_loss)):
            return True

        if take_profit is not None and ((self.side == "buy" and price >= take_profit) or
                                        (self.side == "sell" and price <= take_profit)):
            return True

        if trailing_stop is not None and ((self.side == "buy" and price <= trailing_stop) or
                                          (self.side == "sell" and price >= trailing_stop)):
            return True

        return False

    @property
    def duration_seconds(self) -> int:
        """Return the holding time in seconds, computing it if missing."""
        if self.hold_seconds is not None:
            return self.hold_seconds
        delta = self.closed_at.replace(tzinfo=timezone.utc) - self.opened_at.replace(tzinfo=timezone.utc)
        return int(delta.total_seconds())

    # ----------------------------------------------------------------------
    # Representation
    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<Trade id={self.id} symbol={self.symbol} side={self.side} "
            f"qty={self.quantity} entry={self.entry_price} exit={self.exit_price}>"
        )