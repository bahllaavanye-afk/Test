"""Bot model — declarative trading bot definitions."""
import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, JSON, Integer, DateTime, Numeric
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base
from app.models.base import TimestampMixin


class Bot(Base, TimestampMixin):
    __tablename__ = "bots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    account_id: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(20), default="equity")  # equity|crypto|polymarket
    # Options Alpha-style "desk" grouping
    desk: Mapped[str] = mapped_column(String(32), default="equity")  # equity|crypto|options|futures|fx|commodities|polymarket
    # Signal source: rule_based=pure indicators; ml_signal=ML model output; hybrid=ML gate + rule entry
    signal_source: Mapped[str] = mapped_column(String(16), default="rule_based")
    ml_model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ml_confidence_threshold: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)

    trigger: Mapped[dict] = mapped_column(JSON, nullable=False)
    conditions: Mapped[list] = mapped_column(JSON, default=list)
    condition_logic: Mapped[str] = mapped_column(String(8), default="ALL")  # ALL | ANY
    action: Mapped[dict] = mapped_column(JSON, nullable=False)
    exit_rules: Mapped[list] = mapped_column(JSON, default=list)

    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_signal: Mapped[str | None] = mapped_column(String(16), nullable=True)  # buy|sell|hold|alert
    last_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    template_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
