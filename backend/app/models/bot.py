"""Bot model — declarative trading bot definitions."""
import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, JSON, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column, validates
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
    exit_rules: Mapped[list] = mapped_column(JSON, default=list)

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

    @validates("trigger")
    def _validate_trigger(self, key: str, value: dict | None) -> dict:
        """Ensure trigger is always a dict; treat None as empty dict."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("trigger must be a dict")
        return value

    @validates("action")
    def _validate_action(self, key: str, value: dict | None) -> dict:
        """Ensure action is always a dict; treat None as empty dict."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("action must be a dict")
        return value

    @validates("conditions")
    def _validate_conditions(self, key: str, value: list | None) -> list:
        """Normalize conditions to an empty list when None or non‑list."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("conditions must be a list")
        return value

    @validates("exit_rules")
    def _validate_exit_rules(self, key: str, value: list | None) -> list:
        """Normalize exit_rules to an empty list when None or non‑list."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("exit_rules must be a list")
        return value

    @validates("market_type")
    def _validate_market_type(self, key: str, value: str | None) -> str:
        """Validate market_type against known types; default to 'equity' on invalid input."""
        if not value or value not in MARKET_TYPES:
            return "equity"
        return value

    @validates("run_count")
    def _validate_run_count(self, key: str, value: int | None) -> int:
        """Guard against negative or None run counts."""
        if value is None or value < 0:
            return 0
        return value

    @validates("last_signal")
    def _validate_last_signal(self, key: str, value: str | None) -> str | None:
        """Allow only known signal values or None."""
        allowed = {"buy", "sell", "hold", "alert"}
        if value is None:
            return None
        if value not in allowed:
            raise ValueError(f"last_signal must be one of {allowed}")
        return value