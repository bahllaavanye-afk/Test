import uuid
from sqlalchemy import String, ForeignKey, Boolean, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin


class Strategy(Base, TimestampMixin):
    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str | None] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)          # e.g. 'pairs_trading'
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    market_type: Mapped[str] = mapped_column(String(16), nullable=False)   # equity|crypto|polymarket
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False) # manual|ml_enhanced
    risk_bucket: Mapped[str] = mapped_column(String(16), nullable=False)   # arbitrage|directional
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    symbols: Mapped[list] = mapped_column(JSON, default=list)              # tracked symbols
    tick_interval_seconds: Mapped[float] = mapped_column(Float, default=60.0)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.60)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    account: Mapped["Account"] = relationship("Account", back_populates="strategies")
