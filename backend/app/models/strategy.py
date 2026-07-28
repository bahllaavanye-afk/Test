import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import String, ForeignKey, Boolean, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from app.database import Base
from app.models.base import TimestampMixin


class Strategy(Base, TimestampMixin):
    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. 'pairs_trading'
    display_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    market_type: Mapped[str] = mapped_column(String(16), nullable=False)   # equity|crypto|polymarket
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False)  # manual|ml_enhanced
    risk_bucket: Mapped[str] = mapped_column(String(16), nullable=False)   # arbitrage|directional
    params: Mapped[Dict] = mapped_column(JSON, default=dict)
    symbols: Mapped[List] = mapped_column(JSON, default=list)              # tracked symbols
    tick_interval_seconds: Mapped[float] = mapped_column(Float, default=60.0)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.60)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    account: Mapped["Account"] = relationship("Account", back_populates="strategies")

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------

    @validates("name")
    def _validate_name(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Strategy name must be a non‑empty string.")
        if len(value) > 64:
            raise ValueError("Strategy name cannot exceed 64 characters.")
        return value

    @validates("display_name")
    def _validate_display_name(self, key: str, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not isinstance(value, str):
            raise ValueError("Display name must be a string if provided.")
        if len(value) > 128:
            raise ValueError("Display name cannot exceed 128 characters.")
        return value

    @validates("market_type")
    def _validate_market_type(self, key: str, value: str) -> str:
        allowed = {"equity", "crypto", "polymarket"}
        if value not in allowed:
            raise ValueError(f"market_type must be one of {allowed}, got '{value}'.")
        return value

    @validates("strategy_type")
    def _validate_strategy_type(self, key: str, value: str) -> str:
        allowed = {"manual", "ml_enhanced"}
        if value not in allowed:
            raise ValueError(f"strategy_type must be one of {allowed}, got '{value}'.")
        return value

    @validates("risk_bucket")
    def _validate_risk_bucket(self, key: str, value: str) -> str:
        allowed = {"arbitrage", "directional"}
        if value not in allowed:
            raise ValueError(f"risk_bucket must be one of {allowed}, got '{value}'.")
        return value

    @validates("tick_interval_seconds")
    def _validate_tick_interval_seconds(self, key: str, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError("tick_interval_seconds must be a numeric type.")
        if value <= 0:
            raise ValueError("tick_interval_seconds must be positive.")
        return float(value)

    @validates("confidence_threshold")
    def _validate_confidence_threshold(self, key: str, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError("confidence_threshold must be a numeric type.")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1 inclusive.")
        return float(value)

    @validates("is_enabled")
    def _validate_is_enabled(self, key: str, value: bool) -> bool:
        if not isinstance(value, bool):
            raise ValueError("is_enabled must be a boolean.")
        return value

    @validates("params")
    def _validate_params(self, key: str, value: Any) -> Dict:
        if not isinstance(value, dict):
            raise ValueError("params must be a dictionary.")
        return value

    @validates("symbols")
    def _validate_symbols(self, key: str, value: Any) -> List:
        if not isinstance(value, list):
            raise ValueError("symbols must be a list.")
        return value