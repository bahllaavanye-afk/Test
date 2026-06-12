"""Strategy promotion pipeline: paper → shadow → staging → live"""
import uuid
from sqlalchemy import String, Float, Integer, Boolean, JSON, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin


class StrategyPromotion(Base, TimestampMixin):
    __tablename__ = "strategy_promotions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_id: Mapped[str] = mapped_column(String, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(16), nullable=False, default="paper")
    # stages: paper | shadow | staging | live | rejected

    # Metrics gathered in each stage (JSON: {sharpe, sortino, max_drawdown, win_rate, num_trades, days_in_stage})
    paper_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    shadow_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    staging_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    live_metrics: Mapped[dict] = mapped_column(JSON, default=dict)

    # Stage timestamps
    paper_started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    shadow_started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    staging_started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    live_started_at: Mapped[str | None] = mapped_column(String, nullable=True)

    # Review state
    last_review_at: Mapped[str | None] = mapped_column(String, nullable=True)
    promotion_ready: Mapped[bool] = mapped_column(Boolean, default=False)  # set True by holistic review
    promotion_ready_stage: Mapped[str | None] = mapped_column(String, nullable=True)  # which stage it's ready to advance to
    awaiting_approval: Mapped[bool] = mapped_column(Boolean, default=False)  # user pinged, waiting for approval

    # Approval/rejection
    approved_by: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_at: Mapped[str | None] = mapped_column(String, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Review history (list of {ts, stage, metrics, verdict, reason})
    review_history: Mapped[list] = mapped_column(JSON, default=list)

    # Notes
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
