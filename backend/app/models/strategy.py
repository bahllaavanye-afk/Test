import uuid
from sqlalchemy import String, ForeignKey, Boolean, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin

# Column length constants
STRATEGY_NAME_MAX_LEN = 64
DISPLAY_NAME_MAX_LEN = 128
MARKET_TYPE_MAX_LEN = 16
STRATEGY_TYPE_MAX_LEN = 16
RISK_BUCKET_MAX_LEN = 16

# Default value constants
DEFAULT_TICK_INTERVAL_SECONDS = 60.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

# String constants
STRATEGY_TABLE_NAME = "strategies"
ACCOUNT_TABLE_NAME = "accounts"
ACCOUNT_ID_COLUMN = "account_id"
ID_COLUMN = "id"
NAME_COLUMN = "name"
DISPLAY_NAME_COLUMN = "display_name"
MARKET_TYPE_COLUMN = "market_type"
STRATEGY_TYPE_COLUMN = "strategy_type"
RISK_BUCKET_COLUMN = "risk_bucket"
PARAMS_COLUMN = "params"
SYMBOLS_COLUMN = "symbols"
TICK_INTERVAL_SECONDS_COLUMN = "tick_interval_seconds"
CONFIDENCE_THRESHOLD_COLUMN = "confidence_threshold"
IS_ENABLED_COLUMN = "is_enabled"

# Relationship constant
ACCOUNT_BACK_POPULATES = "strategies"


class Strategy(Base, TimestampMixin):
    __tablename__ = STRATEGY_TABLE_NAME

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str | None] = mapped_column(String, ForeignKey(f"{ACCOUNT_TABLE_NAME}.{ID_COLUMN}", ondelete="CASCADE"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(STRATEGY_NAME_MAX_LEN), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(DISPLAY_NAME_MAX_LEN), nullable=True)
    market_type: Mapped[str] = mapped_column(String(MARKET_TYPE_MAX_LEN), nullable=False)
    strategy_type: Mapped[str] = mapped_column(String(STRATEGY_TYPE_MAX_LEN), nullable=False)
    risk_bucket: Mapped[str] = mapped_column(String(RISK_BUCKET_MAX_LEN), nullable=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    symbols: Mapped[list] = mapped_column(JSON, default=list)
    tick_interval_seconds: Mapped[float] = mapped_column(Float, default=DEFAULT_TICK_INTERVAL_SECONDS)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    account: Mapped["Account"] = relationship("Account", back_populates=ACCOUNT_BACK_POPULATES)